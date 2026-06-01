"""
This script implements the test time adaptation for performing FFS/FMS.
"""

import json
import os

import torch
import pandas as pd
import numpy as np
import pyvista as pv
from torch.utils.data import DataLoader, Subset
from argparse import ArgumentParser
from tqdm import tqdm
from skimage import measure
from deformation.deformation_attack import Deformer
from utils.utils import set_random_seed, save_torch_to_nifti
from data_utils.data import SkullDataset
from data_utils.data_utils import split_dataset_patient_level
from classifier.ResNet import resnet18, resnet34, resnet50, resnet101
from classifier.SeResNeXt import seresnet3d_18, seresnet3d_34, seresnet3d_50, seresnet3d_101


# Maps a model name to its constructor and the checkpoint path. Update each
# path to point at your own trained classifier weights (one per architecture).
MODEL_REGISTRY = {
    'resnet18':    (resnet18,       'path/to/resnet18.pth'),
    'resnet34':    (resnet34,       'path/to/resnet34.pth'),
    'resnet50':    (resnet50,       'path/to/resnet50.pth'),
    'resnet101':   (resnet101,      'path/to/resnet101.pth'),
    'seresnet18':  (seresnet3d_18,  'path/to/seresnet18.pth'),
    'seresnet34':  (seresnet3d_34,  'path/to/seresnet34.pth'),
    'seresnet50':  (seresnet3d_50,  'path/to/seresnet50.pth'),
    'seresnet101': (seresnet3d_101, 'path/to/seresnet101.pth'),
}


def _parse_model_list(spec):
    names = [n.strip() for n in spec.split(',') if n.strip()]
    unknown = [n for n in names if n not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown model name(s): {unknown}. "
            f"Known: {sorted(MODEL_REGISTRY)}"
        )
    return names


def _load_model(name, device):
    constructor, ckpt_path = MODEL_REGISTRY[name]
    model = constructor(in_channels=1, num_classes=1)
    state = torch.load(ckpt_path, weights_only=False)['model_state_dict']
    model.load_state_dict(state)
    model.to(device)
    return model


def main(args):
    """
    Main function to call for performing classifier guided image deformation.
    :param args: parameters parsed from the command line.
    :return: Nothing.
    """

    """ Set a device to use """
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    """ Enable determinism """
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """ Dataset """
    dataset = SkullDataset(data_dir=args.data_dir, metadata_csv=args.metadata_csv)
    train_indices, val_indices, test_indices = split_dataset_patient_level(dataset, val_fraction=0.15,
                                                                           test_fraction=0.15)
    test_dataset = Subset(SkullDataset(args.data_dir, args.metadata_csv, augment=False), test_indices)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    out_dir = os.path.join('./deformed_images', args.exp_name)
    for sub in ('input_male', 'input_female', 'gen_male', 'gen_female'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    """ Load classification models """
    opt_names = _parse_model_list(args.opt_models)
    eval_names = _parse_model_list(args.eval_models)

    overlap = set(opt_names) & set(eval_names)
    if overlap:
        raise ValueError(
            f"Models {sorted(overlap)} appear in both --opt_models and --eval_models; "
            f"eval classifiers must be held out from optimization."
        )

    print(f"[INFO] Experiment: {args.exp_name}")
    print(f"[INFO] Optimization models ({len(opt_names)}): {opt_names}")
    print(f"[INFO] Held-out eval models ({len(eval_names)}): {eval_names}")

    model_cache = {n: _load_model(n, device) for n in (opt_names + eval_names)}
    model_list = [model_cache[n] for n in opt_names]
    eval_model_list = [model_cache[n] for n in eval_names]

    results = []

    n_samples = len(test_loader)
    trajectories = {} if args.save_logit_trajectory else None
    trajectories_path = (
        os.path.join(out_dir, f"trajectories_{args.exp_name}.json")
        if args.save_logit_trajectory else None
    )

    for i, data in enumerate(tqdm(test_loader, desc="Samples")):
        image, label = data['image'], data['sex']
        target_class = 0 if label == 1 else 1

        deformer = Deformer(
            img_size=args.img_size,
            cp_spacing=args.cps,
            model_list=model_list,
            lr=args.lr,
            device=device,
            margin=args.logit_margin,
        )

        def_img = deformer.deform(
            img=image,
            target_class=target_class,
            steps=args.steps,
            log_step=args.print_step,
            record_trajectory=args.save_logit_trajectory,
        )

        if args.save_logit_trajectory:
            patient_id_str = data['patient_id'][0]
            trajectories[i] = {
                'patient_id': patient_id_str,
                'target_class': int(target_class),
                **deformer.last_trajectory,
            }
            with open(trajectories_path, "w") as f:
                json.dump(trajectories, f, indent=2)

        for model in model_list:
            model.to('cpu')

        for model in eval_model_list:
            model.to(device)

        with torch.no_grad():
            def_img_class = def_img[:, :, :, 128:, :]
            ensemble_eval_logits = []
            for j, model in enumerate(eval_model_list):
                logits = model(def_img_class)
                ensemble_eval_logits.append(logits)

            individual_logits = [l.item() for l in ensemble_eval_logits]
            individual_probs = [torch.sigmoid(l).item() for l in ensemble_eval_logits]
            average_prob = sum(individual_probs) / len(individual_probs)

            jac_stats = deformer.jacobian_stats()

            patient_id = data['patient_id'][0]
            direction = "M→F" if target_class == 1 else "F→M"
            worst_logit = min(individual_logits) if target_class == 1 else max(individual_logits)
            tqdm.write(
                f"[{i+1:>3}/{n_samples}] pid={patient_id} {direction} | "
                f"worst={worst_logit:+.2f}  μ_p={average_prob:.3f} | "
                f"det_J ∈ [{jac_stats['det_J_min']:.2f}, {jac_stats['det_J_max']:.2f}] "
                f"(μ={jac_stats['det_J_mean']:.2f}, "
                f"neg={jac_stats['det_J_neg_fraction']*100:.2f}%)"
            )
            result_row = {
                'idx': i,
                'patient_id': patient_id,
                'orig_label': int(label.item()),
                'target_class': int(target_class),
                'avg_prob_after': average_prob,
                'individual_probs': individual_probs,
                'individual_logits': individual_logits,
                **jac_stats,
            }
            for name, prob, logit in zip(eval_names, individual_probs, individual_logits):
                result_row[f'prob_{name}'] = prob
                result_row[f'logit_{name}'] = logit
            results.append(result_row)

            orig_sex_dir = 'input_female' if int(label.item()) == 1 else 'input_male'
            gen_sex_dir = 'gen_female' if target_class == 1 else 'gen_male'

            input_save_path = os.path.join(out_dir, orig_sex_dir,
                                           f"input_{i:04d}_pid_{patient_id}.nii.gz")
            save_torch_to_nifti(image.squeeze().detach().cpu(), input_save_path)
            input_img_mesh = image.squeeze().detach().cpu().numpy().astype(np.float32)
            try:
                verts, faces, _, _ = measure.marching_cubes(input_img_mesh, level=0.5, spacing=(1.0, 1.0, 1.0))
                faces_pv = np.hstack([
                    np.full((len(faces), 1), 3, dtype=np.int32),
                    faces.astype(np.int32)
                ]).ravel()
                mesh = pv.PolyData(verts, faces_pv)
                mesh = mesh.clean()
                mesh = mesh.smooth(n_iter=3)
                mesh = mesh.compute_normals(auto_orient_normals=True)

                mesh.save(os.path.join(out_dir, orig_sex_dir,
                                       f"input_{i:04d}_pid_{patient_id}.ply"))
            except (ValueError, RuntimeError) as e:
                print(f"Warning: could not generate mesh: {e}")

            save_path = os.path.join(out_dir, gen_sex_dir,
                                     f"sample_{i:04d}_pid_{patient_id}.nii.gz")
            save_torch_to_nifti(def_img.squeeze().detach().cpu(), save_path)
            def_img_mesh = def_img.squeeze().detach().cpu().numpy().astype(np.float32)
            try:
                verts, faces, _, _ = measure.marching_cubes(def_img_mesh, level=0.5, spacing=(1.0, 1.0, 1.0))
                faces_pv = np.hstack([
                    np.full((len(faces), 1), 3, dtype=np.int32),
                    faces.astype(np.int32)
                ]).ravel()
                mesh = pv.PolyData(verts, faces_pv)
                mesh = mesh.clean()
                mesh = mesh.smooth(n_iter=3)
                mesh = mesh.compute_normals(auto_orient_normals=True)
                mesh.save(os.path.join(out_dir, gen_sex_dir,
                                       f"sample_{i:04d}_pid_{patient_id}.ply"))
            except (ValueError, RuntimeError) as e:
                print(f"Warning: could not generate mesh: {e}")

            for model in eval_model_list:
                model.to('cpu')

            for model in model_list:
                model.to(device)

    df = pd.DataFrame(results)
    out_path = os.path.join(out_dir, f'eval_{args.exp_name}.csv')
    df.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved detailed deformation results to {out_path}")

    # Compute summary stats
    print("\n===== Deformation Evaluation Summary =====")
    print(df.groupby('orig_label')['avg_prob_after'].describe())


if __name__ == "__main__":
    def parse_args():
        parser = ArgumentParser()

        """ General """
        parser.add_argument('--gpu_id', type=int, default=0)
        parser.add_argument('--seed', type=int, default=42)

        parser.add_argument('--data_dir', type=str,
                            default=os.environ.get('AUTOFFS_DATA_DIR', './data'))
        parser.add_argument('--metadata_csv', type=str,
                            default=os.environ.get('AUTOFFS_METADATA_CSV', './data/metadata.csv'))

        parser.add_argument('-steps', type=int, default=100)

        """ Experiment """
        parser.add_argument(
            '--exp_name',
            type=str,
            required=True,
            help='Name of this experiment; outputs go to ./deformed_images/{exp_name}/',
        )
        parser.add_argument(
            '--opt_models',
            type=str,
            default='resnet18,resnet34,seresnet34,resnet50,seresnet50,seresnet101',
            help=(
                'Comma-separated list of classifier names used for deformation '
                'optimization. Choices: ' + ','.join(sorted(MODEL_REGISTRY)) + '.'
            ),
        )
        parser.add_argument(
            '--eval_models',
            type=str,
            default='seresnet18,resnet101',
            help='Comma-separated list of held-out classifiers used to score the transformed images.',
        )

        """ Logging """
        parser.add_argument('--print_step', type=int, default=10)
        parser.add_argument(
            '--save_logit_trajectory',
            action='store_true',
            help=('Record per-step worst logit / mean prob / losses for every sample '
                  'and write to ./deformed_images/{exp_name}/trajectories_{exp_name}.json. '
                  'File is rewritten after each sample, so a Ctrl-C keeps completed samples.'),
        )

        """ Dataset """
        parser.add_argument('--img_size', type=lambda s: tuple(int(x) for x in s.split(',')),
                            default=(256, 256, 256),
                            help='Volume size as comma-separated ints, e.g. 256,256,256.')

        """ Optimizer """
        parser.add_argument('--lr', type=float, default=5e-3)

        """ Deformation """
        parser.add_argument('--cps', help='Spacing between control points with respect to image grid', type=int, default=8)

        """ Logit margin for Loss """
        parser.add_argument('--logit_margin', help='Logit margin for SWM Loss', type=float, default=4.5)

        """ Parse Arguments """
        args = parser.parse_args()

        return args

    args = parse_args()
    main(args)