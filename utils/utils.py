import datetime
import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import itertools


def plot_confusion_matrix(cm, class_names):
    """Returns a matplotlib figure containing the plotted confusion matrix."""
    figure = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    threshold = cm_normalized.max() / 2.

    # Add text annotations
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > threshold else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return figure

def log_middle_slices(writer, dataset, tag='train/middle_slices', num_examples=4, global_step=0, device='cpu'):
    n = min(num_examples, len(dataset))
    for i in range(n):
        sample = dataset[i]
        img = sample['image']

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        else:
            img = np.asarray(img)

        # Handle (C,X,Y,Z) -> take first channel, or assume (X,Y,Z)
        if img.ndim == 4:
            img = img[0]

        X, Y, Z = img.shape  # RAS+: (R, A, S) = (256, 128, 256)

        # Extract middle slices
        sagittal = img[X // 2, :, :]  # slice along R-L → (Y, Z) = (A, S)
        coronal = img[:, Y // 2, :]   # slice along A-P → (X, Z) = (R, S)
        axial = img[:, :, Z // 2]     # slice along S-I → (X, Y) = (R, A)

        # Rotate for radiological display (may need adjustment based on preference)
        sagittal = np.rot90(sagittal, k=1)  # A-S plane
        coronal = np.rot90(coronal, k=1)    # R-S plane
        axial = np.rot90(axial, k=1)        # R-A plane

        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        cmap = 'gray'

        axs[0].imshow(sagittal, cmap=cmap, vmin=0, vmax=1)
        axs[0].set_title(f'Sagittal (X={X // 2})')
        axs[0].axis('off')

        axs[1].imshow(coronal, cmap=cmap, vmin=0, vmax=1)
        axs[1].set_title(f'Coronal (Y={Y // 2})')
        axs[1].axis('off')

        axs[2].imshow(axial, cmap=cmap, vmin=0, vmax=1)
        axs[2].set_title(f'Axial (Z={Z // 2})')
        axs[2].axis('off')

        plt.tight_layout()
        writer.add_figure(f"{tag}/example_{i}", fig, global_step=global_step)
        plt.close(fig)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_run_dir(model_type, base_dir='logs'):
    # Get current date and time
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")

    # Build the run directory name
    run_name = f'run_{model_type}_{date_str}_{time_str}'
    run_path = os.path.join(base_dir, run_name)

    # Create directory
    os.makedirs(run_path, exist_ok=True)
    return run_path


def make_coordinate_tensor(dims):
    coordinate_tensor = [torch.linspace(-1, 1, d) for d in dims]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing='ij')
    coordinate_tensor = torch.stack(coordinate_tensor, dim=-1)
    return coordinate_tensor


def save_torch_to_nifti(image, path):
    img_np = image.detach().cpu().squeeze().numpy()
    img_nifti = nib.Nifti1Image(img_np, np.eye(4))
    nib.save(img_nifti, path)
