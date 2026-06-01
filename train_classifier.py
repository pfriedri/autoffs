"""
This script implements binary classifier training with optional masking strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, roc_auc_score,  accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from pathlib import Path

from classifier.ResNet import resnet18, resnet34, resnet50, resnet101
from classifier.SeResNeXt import seresnet3d_18, seresnet3d_34, seresnet3d_50, seresnet3d_101

from data_utils.data import SkullDataset
from data_utils.data_utils import split_dataset_patient_level
from utils.args import parse_args
from utils.utils import create_run_dir, set_random_seed, plot_confusion_matrix, log_middle_slices


# ======================================================================================================================
# Masking Strategies
# ======================================================================================================================
class MaskingStrategy:
    def __init__(self, mask_ratio: float = 0.5):
        self.mask_ratio = mask_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RandomBoxMasking(MaskingStrategy):
    """Randomly mask out cubic/rectangular regions of the input volume."""
    def __init__(self, mask_ratio: float = 0.5, num_masks: int = 1, min_size: float = 0.2, max_size: float = 0.5):

        super().__init__(mask_ratio)
        self.num_masks = num_masks
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        mask = torch.ones_like(x)

        for b in range(B):
            for _ in range(self.num_masks):
                # Random box size
                d_size = int(D * np.random.uniform(self.min_size, self.max_size))
                h_size = int(H * np.random.uniform(self.min_size, self.max_size))
                w_size = int(W * np.random.uniform(self.min_size, self.max_size))

                # Random position
                d_start = np.random.randint(0, max(1, D - d_size))
                h_start = np.random.randint(0, max(1, H - h_size))
                w_start = np.random.randint(0, max(1, W - w_size))

                mask[b, :, d_start:d_start + d_size, h_start:h_start + h_size, w_start:w_start + w_size] = 0

        return x * mask


class RandomPatchMasking(MaskingStrategy):
    """Divide volume into patches and randomly mask a fraction of them."""

    def __init__(self, mask_ratio: float = 0.5, patch_size: tuple = (32, 32, 16)):

        super().__init__(mask_ratio)
        self.patch_size = patch_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        pd, ph, pw = self.patch_size

        # Number of patches in each dimension
        nd = D // pd
        nh = H // ph
        nw = W // pw

        total_patches = nd * nh * nw
        num_masked = int(total_patches * self.mask_ratio)

        mask = torch.ones_like(x)

        for b in range(B):
            # Randomly select patches to mask
            masked_indices = np.random.choice(total_patches, num_masked, replace=False)

            for idx in masked_indices:
                # Convert flat index to 3D patch coordinates
                pz = idx // (nh * nw)
                py = (idx % (nh * nw)) // nw
                px = idx % nw

                d_start, d_end = pz * pd, (pz + 1) * pd
                h_start, h_end = py * ph, (py + 1) * ph
                w_start, w_end = px * pw, (px + 1) * pw

                mask[b, :, d_start:d_end, h_start:h_end, w_start:w_end] = 0

        return x * mask


class SliceMasking(MaskingStrategy):
    """Mask out random slices along specified axes."""

    def __init__(self, mask_ratio: float = 0.3, axes: tuple = (0, 1, 2)):
        super().__init__(mask_ratio)
        self.axes = axes

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        dims = [D, H, W]

        mask = torch.ones_like(x)

        for b in range(B):
            # Randomly choose which axis to mask for this sample
            axis = np.random.choice(self.axes)
            dim_size = dims[axis]
            num_slices_to_mask = int(dim_size * self.mask_ratio)

            slices_to_mask = np.random.choice(dim_size, num_slices_to_mask, replace=False)

            for s in slices_to_mask:
                if axis == 0:
                    mask[b, :, s, :, :] = 0
                elif axis == 1:
                    mask[b, :, :, s, :] = 0
                else:
                    mask[b, :, :, :, s] = 0

        return x * mask


def get_masking_strategy(name: str, **kwargs) -> MaskingStrategy:
    strategies = {
        'random_box': RandomBoxMasking,
        'random_patch': RandomPatchMasking,
        'slice': SliceMasking,
        'none': lambda **kw: None,
    }
    if name not in strategies:
        raise ValueError(f"Unknown masking strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name](**kwargs)

# ======================================================================================================================
# Utils
# ======================================================================================================================
def _compute_auroc(outputs, targets):
    try:
        probs = torch.sigmoid(outputs).cpu().numpy()
        targets_np = targets.cpu().numpy()

        if probs.ndim > 1:
            probs = probs.squeeze()
        if targets_np.ndim > 1:
            targets_np = targets_np.squeeze()
        if len(np.unique(targets_np)) < 2:
            print("[Warning] Only one class present in targets, AUROC undefined.")
            return None
        auroc = roc_auc_score(targets_np, probs)
        return auroc

    except Exception as e:
        print(f"[Warning] Failed to compute AUROC: {e}")
        return None

def get_val_metrics(output, target, threshold=0.5, from_logits=True):
    if from_logits:
        output = torch.sigmoid(output)

    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    y_pred = (output_np >= threshold).astype(int)
    y_true = (target_np >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

    return metrics

# ======================================================================================================================
# Trainer class
# ======================================================================================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, total_epochs, device, writer,
                 log_interval, run_dir, early_stopping_patience, accumulation_steps,
                 masking_strategy=None, mask_prob=0.5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_epochs = total_epochs
        self.device = device
        self.writer = writer
        self.log_interval = log_interval
        self.run_dir = run_dir
        self.accumulation_steps = accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0

        # Masking
        self.masking_strategy = masking_strategy
        self.mask_prob = mask_prob

        self.best_val_metric = float('-inf')
        self.epoch = 0
        self.global_step = 0

    def train(self, num_epochs, save_best='accuracy'):
        for epoch in range(num_epochs):
            self.epoch = epoch + 1

            # Train
            _ = self.train_epoch()

            # Validate (always without masking)
            val_metrics = self.validate()

            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(save_best, 0))
                elif not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('utils/lr', current_lr, global_step=self.epoch)

            # Save best model
            if val_metrics[save_best] >= self.best_val_metric:
                self.best_val_metric = val_metrics[save_best]
                self.save_checkpoint('best_model.pth', val_metrics)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                print(
                    f"No improvement in '{save_best}' for {self.early_stopping_counter}/{self.early_stopping_patience} epochs.")

            if not self.early_stopping_patience == 0 and self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {self.epoch} epochs. "
                      f"Best {save_best}: {self.best_val_metric:.4f}")
                break

        # Save last model
        self.save_checkpoint('last_model.pth', val_metrics)

    def train_epoch(self):
        accumulated_loss = 0.0
        self.model.train()
        self.optimizer.zero_grad()

        num_masked_batches = 0
        total_batches = 0

        pbar = tqdm(self.train_loader, desc=f'[Train] Epoch ({self.epoch}/{self.total_epochs})')

        # Train model with gradient accumulation
        for batch_idx, data in enumerate(pbar):
            img, target = data['image'].to(self.device), data['sex'].to(self.device).unsqueeze(dim=-1).float()

            # Apply masking with probability mask_prob
            if self.masking_strategy is not None and np.random.random() < self.mask_prob:
                img = self.masking_strategy(img)
                num_masked_batches += 1
            total_batches += 1

            output = self.model(img)
            loss = self.criterion(output, target)
            loss = loss / self.accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

                self.global_step += 1

                if self.global_step % self.log_interval == 0:
                    self.writer.add_scalar('training/criterion', accumulated_loss, global_step=self.global_step)
                    pbar.set_postfix(**{'criterion': accumulated_loss})

                accumulated_loss = 0.0

        # Handle leftover batches
        if (batch_idx + 1) % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        # Log masking statistics
        if self.masking_strategy is not None and self.writer is not None:
            mask_ratio = num_masked_batches / total_batches if total_batches > 0 else 0
            self.writer.add_scalar('training/mask_ratio', mask_ratio, global_step=self.epoch)

        return accumulated_loss

    def validate(self):
        self.model.eval()
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'[Validation] Epoch ({self.epoch}/{self.total_epochs})')

            for data in pbar:
                img, target = data['image'].to(self.device), data['sex'].to(self.device).unsqueeze(dim=-1).float()

                output = self.model(img)
                all_outputs.append(output)
                all_targets.append(target)

            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Compute validation metrics
            val_metrics = get_val_metrics(all_outputs, all_targets, from_logits=True)
            auroc = _compute_auroc(all_outputs, all_targets)
            if auroc is not None:
                val_metrics['auroc'] = auroc

            # Log all metrics
            self.writer.add_scalar('validation/accuracy', val_metrics['accuracy'], global_step=self.epoch)
            self.writer.add_scalar('validation/f1', val_metrics['f1'], global_step=self.epoch)
            if auroc is not None:
                self.writer.add_scalar('validation/auroc', auroc, global_step=self.epoch)
            pbar.set_postfix(val_metrics)

            # Confusion matrix
            try:
                preds = (torch.sigmoid(all_outputs) > 0.5).cpu().numpy().astype(int)
                targets = all_targets.cpu().numpy().astype(int)

                cm = confusion_matrix(targets, preds)
                class_names = ['Male', 'Female']
                fig = plot_confusion_matrix(cm, class_names)

                self.writer.add_figure('validation/confusion_matrix', fig, global_step=self.epoch)
                plt.close(fig)
            except Exception as e:
                print(f"[Warning] Failed to plot confusion matrix: {e}")

        return val_metrics

    def test(self):
        self.model.eval()
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='[Testing]')

            for data in pbar:
                img, target = data['image'].to(self.device), data['sex'].to(self.device).unsqueeze(dim=-1).float()

                output = self.model(img)
                all_outputs.append(output)
                all_targets.append(target)

            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            preds = (torch.sigmoid(all_outputs) > 0.5).cpu().numpy().astype(int)
            targets = all_targets.cpu().numpy().astype(int)

            test_metrics = {}
            test_metrics['accuracy'] = accuracy_score(targets, preds.squeeze())
            test_metrics['f1'] = f1_score(targets, preds.squeeze())
            auroc = _compute_auroc(all_outputs, all_targets)
            if auroc is not None:
                test_metrics['auroc'] = auroc

            # Log test metrics
            if self.writer:
                self.writer.add_scalar('test/accuracy', test_metrics['accuracy'], global_step=self.epoch)
                self.writer.add_scalar('test/f1', test_metrics['f1'], global_step=self.epoch)
                if auroc is not None:
                    self.writer.add_scalar('test/auroc', auroc, global_step=self.epoch)

                try:
                    cm = confusion_matrix(targets, preds)
                    class_names = ['Male', 'Female']
                    fig = plot_confusion_matrix(cm, class_names)
                    self.writer.add_figure('test/confusion_matrix', fig, global_step=self.epoch)
                    plt.close(fig)
                except Exception as e:
                    print(f"[Warning] Failed to plot test confusion matrix: {e}")

            print(f"\n{'=' * 50}")
            for metric, value in test_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
            print(f"{'=' * 50}\n")

        return test_metrics

    def save_checkpoint(self, filename, val_metrics):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': val_metrics,
            'masking_strategy': type(self.masking_strategy).__name__ if self.masking_strategy else None,
            'mask_prob': self.mask_prob,
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.run_dir / filename)
        print(f'Saved checkpoint {filename}')

    def load_checkpoint(self, filename):
        checkpoint = torch.load(self.run_dir / filename, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint.get('epoch', 0)

        print(f'Loaded checkpoint {filename} (Epoch {self.epoch}')

        return checkpoint.get('metrics', {})


# ======================================================================================================================
# Main function
# ======================================================================================================================
def main():
    # Get config
    args = parse_args()

    # Set determinism
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Setup new run directory
    log_dir = create_run_dir(args.model)

    # Setup Tensorboard logging
    writer = None
    if args.log_tensorboard:
        writer = SummaryWriter(log_dir=log_dir)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get dataloaders
    dataset = SkullDataset(data_dir=args.data_dir, metadata_csv=args.metadata_csv, front_only=True)
    train_indices, val_indices, test_indices = split_dataset_patient_level(dataset,
                                                                           val_fraction=0.15,
                                                                           test_fraction=0.15)

    train_dataset = Subset(SkullDataset(args.data_dir, args.metadata_csv, augment=True, front_only=True), train_indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, prefetch_factor=4)

    val_dataset = Subset(SkullDataset(args.data_dir, args.metadata_csv, augment=False, front_only=True), val_indices)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    test_dataset = Subset(SkullDataset(args.data_dir, args.metadata_csv, augment=False, front_only=True), test_indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # Calculate accumulation steps from effective batch size (for gradient accumulation)
    accumulation_steps = args.effective_batch_size // args.batch_size
    print(f"[Training] batch_size={args.batch_size}, accumulation_steps={accumulation_steps}, "
          f"effective_batch_size={args.batch_size * accumulation_steps}")

    # Determine class weights for unbalanced data
    data = train_dataset.dataset.data.iloc[train_dataset.indices]
    num_male = (data['sex'] == 'M').sum()
    num_female = (data['sex'] == 'F').sum()
    pos_weight = torch.tensor([num_male / num_female]).to(device)
    print(f"[Class Balance] Male: {num_male}, Female: {num_female}, pos_weight: {pos_weight.item():.3f}")

    # Define criterion
    if args.criterion == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        print(f"[Criterion] BCEWithLogitsLoss (unweighted)")
    elif args.criterion == 'bce_weighted':
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[Criterion] BCEWithLogitsLoss (pos_weight={pos_weight.item():.3f})")
    else:
        raise ValueError(f'Criterion {args.criterion} not found. Available: bce')

    # Setup masking strategy
    masking_strategy = None
    mask_prob = getattr(args, 'mask_prob', 0.5)
    masking_type = getattr(args, 'masking', 'none')

    if masking_type != 'none':
        # Get masking-specific parameters from args
        masking_kwargs = {}

        if masking_type == 'random_box':
            masking_kwargs = {
                'num_masks': getattr(args, 'num_masks', 2),
                'min_size': getattr(args, 'mask_min_size', 0.2),
                'max_size': getattr(args, 'mask_max_size', 0.4),
            }
        elif masking_type == 'random_patch':
            masking_kwargs = {
                'mask_ratio': getattr(args, 'mask_ratio', 0.4),
                'patch_size': tuple(getattr(args, 'patch_size', [64, 32, 64])),
            }
        elif masking_type == 'slice':
            masking_kwargs = {
                'mask_ratio': getattr(args, 'mask_ratio', 0.3),
            }

        masking_strategy = get_masking_strategy(masking_type, **masking_kwargs)
        print(f"[Masking] Using {masking_type} masking with prob={mask_prob}")
        print(f"[Masking] Parameters: {masking_kwargs}")
    else:
        print("[Masking] No masking strategy will be used.")

    # Create model
    # ----- (4) 3D ResNets [resnet18, resnet34, resnet50, resnet101] -----
    if args.model == 'resnet18':
        model = resnet18(in_channels=args.in_channel, num_classes=args.num_classes).to(device)
    elif args.model == 'resnet34':
        model = resnet34(in_channels=args.in_channel, num_classes=args.num_classes).to(device)
    elif args.model == 'resnet50':
        model = resnet50(in_channels=args.in_channel, num_classes=args.num_classes).to(device)
    elif args.model == 'resnet101':
        model = resnet101(in_channels=args.in_channel, num_classes=args.num_classes).to(device)

    # ----- (4) 3D SE-ResNets [seresnet18, seresnet34, seresnet50, seresnet101] -----
    elif args.model == 'seresnet18':
        model = seresnet3d_18(in_channels=args.in_channel, num_classes=args.num_classes).to(device)
    elif args.model == 'seresnet34':
        model = seresnet3d_34(in_channels=args.in_channel, num_classes=args.num_classes).to(device)
    elif args.model == 'seresnet50':
        model = seresnet3d_50(in_channels=args.in_channel, num_classes=args.num_classes).to(device)
    elif args.model == 'seresnet101':
        model = seresnet3d_101(in_channels=args.in_channel, num_classes=args.num_classes).to(device)

    else:
        raise ValueError(f'Model {args.model} not found.')

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Create scheduler ['none', 'plateau', 'onecycle']
    scheduler = None
    if args.scheduler is not None:
        if args.scheduler == 'plateau':
            mode = 'max' if args.save_best in ['accuracy', 'f1', 'auroc'] else 'min'
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=5,
                                                                   factor=0.5, min_lr=1e-6)
            print(f"[Scheduler] Using ReduceLROnPlateau")

        elif args.scheduler == 'onecycle':
            steps_per_epoch = max(1, len(train_loader) // accumulation_steps)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.num_epochs,
                                                            steps_per_epoch=steps_per_epoch, pct_start=0.05,
                                                            anneal_strategy='cos', div_factor=10, final_div_factor=100)
            print(f"[Scheduler] Using OneCycleLR")

        else:
            raise ValueError(f'Scheduler {args.scheduler} not found. Available: plateau, onecycle')
    else:
        print("[Scheduler] No scheduler will be used.")

    # Log example data (with and without masking for comparison)
    if writer is not None:
        log_middle_slices(writer, train_dataset, tag='train/middle_slices', num_examples=3, global_step=0,
                          device=device)

        # Log masked examples if masking is enabled
        if masking_strategy is not None:
            sample_data = next(iter(train_loader))
            sample_img = sample_data['image'].to(device)
            masked_img = masking_strategy(sample_img)

            # Log a few slices showing the masking effect
            for i in range(min(3, sample_img.shape[0])):
                mid_slice = sample_img.shape[2] // 2

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(sample_img[i, 0, mid_slice].cpu().numpy(), cmap='gray')
                axes[0].set_title('Original')
                axes[0].axis('off')

                axes[1].imshow(masked_img[i, 0, mid_slice].cpu().numpy(), cmap='gray')
                axes[1].set_title('Masked')
                axes[1].axis('off')

                writer.add_figure(f'masking/example_{i}', fig, global_step=0)
                plt.close(fig)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        total_epochs=args.num_epochs,
        device=device,
        writer=writer,
        log_interval=args.log_interval,
        run_dir=Path(log_dir),
        early_stopping_patience=args.early_stopping_patience,
        accumulation_steps=accumulation_steps,
        masking_strategy=masking_strategy,
        mask_prob=mask_prob,
    )

    # Train model
    trainer.train(
        num_epochs=args.num_epochs,
        save_best=args.save_best,
    )

    # Test model
    print("\nLoading best model for testing...")
    trainer.load_checkpoint('best_model.pth')

    trainer.val_loader = test_loader
    test_metrics = trainer.test()

    with open(Path(log_dir) / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Test results saved to {Path(log_dir) / 'test_results.json'}")

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
