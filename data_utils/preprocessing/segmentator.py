"""
Script for segmenting T1-weighted skull MRI.
We apply a pretrained segmentation model from: https://github.com/lab-smile/GRACE
"""

import os
import glob
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    EnsureTyped,
    ScaleIntensityRangePercentilesd
)
from monai.data import DataLoader, Dataset
from tqdm import tqdm


def load_data_files(data_dir):
    nii_files = sorted(glob.glob(os.path.join(data_dir, "**/*t1n_3d.nii.gz"), recursive=True))
    print(f"Found {len(nii_files)} T1-weighted .nii.gz files")
    if len(nii_files) == 0:
        raise ValueError(f"No T1-weighted files found in {data_dir}")
    files = [{"image": file_path} for file_path in nii_files]
    return files


def create_preprocessing_transforms():
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),  # Resample to 1mm isotropic voxels
            mode="bilinear"
        ),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,  # 1st percentile
            upper=99,  # 99th percentile
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        EnsureTyped(keys=["image"]),
    ])
    return transforms


def load_segmentation_model(model_path, device):
    model = nn.DataParallel(
        UNETR(
            in_channels=1,  # Single-channel grayscale input
            out_channels=12,  # 12 segmentation classes
            img_size=(64, 64, 64),  # Patch size for sliding window
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ),
        device_ids=[0]
    ).cuda()

    # Load pre-trained weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def perform_inference(model, image_tensor, patch_size=(64, 64, 64), overlap=0.8):
    input_tensor = torch.unsqueeze(image_tensor, 1).cuda()
    output = sliding_window_inference(
        inputs=input_tensor,
        roi_size=patch_size,
        sw_batch_size=32,
        predictor=model,
        overlap=overlap
    )
    return output


def save_segmentation(output_tensor, reference_image_path, save_path):
    segmentation = torch.argmax(output_tensor, dim=1).detach().cpu().numpy()[0]

    ref_nii = nib.load(reference_image_path)
    ref_header = ref_nii.header.copy()
    ref_affine = ref_nii.affine

    new_img = nib.Nifti1Image(
        segmentation.astype(np.uint8),
        affine=ref_affine,
        header=ref_header
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    nib.save(new_img, save_path)


def main():
    DATA_DIR = 'path/to/your/data/'
    MODEL_PATH = 'path/to/your/model.pth'

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    files = load_data_files(DATA_DIR)

    # Create preprocessing pipeline
    transforms = create_preprocessing_transforms()

    # Create dataset
    dataset = Dataset(data=files, transform=transforms)

    # Load model
    model = load_segmentation_model(MODEL_PATH, device)

    print("\nStarting segmentations...")
    for i in tqdm(range(len(dataset)), desc="Segmenting images"):
        with torch.no_grad():
            # Get image data
            data = dataset[i]
            img = data["image"]
            img_path = img.meta["filename_or_obj"]

            # Perform inference
            output = perform_inference(model, img)

            # Prepare save path
            # Extract path before 't1n' and append 'segmentation.nii.gz'
            base_path = img_path.split("/t1n")[0]
            save_path = os.path.join(base_path, "segmentation.nii.gz")

            # Save segmentation
            save_segmentation(output, img_path, save_path)
            tqdm.write(f"Saved: {save_path}")

    print("\nSegmentations complete!")

if __name__ == '__main__':
    main()
