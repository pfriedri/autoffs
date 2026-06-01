import numpy as np
import torchio as tio


def load_and_process_data(file_path, target_size=(256, 256, 256), augment=False):
    # Load image
    subject = tio.Subject(mask=tio.LabelMap(file_path))
    preprocessing = tio.Compose([
        tio.ToCanonical(),
        tio.CropOrPad(target_size),
    ])
    subject = preprocessing(subject)

    if augment:
        augmentation = tio.Compose([
           tio.RandomAffine(scales=(0.9, 1.10), degrees=10, translation=10),
           tio.RandomFlip(axes='LR', flip_probability=0.5),
        ])
        subject = augmentation(subject)

    return subject['mask'].data.numpy().astype(np.int16)

def augment_sdf(sdf_volume):
    """Apply augmentations to precomputed SDF."""
    subject = tio.Subject(sdf=tio.ScalarImage(tensor=sdf_volume))

    transform = tio.Compose([
        tio.RandomAffine(scales=(0.90, 1.10), degrees=5, translation=10, default_pad_value='mean'),
        tio.RandomFlip(axes='LR', flip_probability=0.5),
    ])

    subject = transform(subject)
    return subject['sdf'].data

def split_dataset_patient_level(dataset, val_fraction, test_fraction):
    """Split the dataset by patients to prevent data leakage."""
    patient_ids = dataset.get_patient_ids()
    np.random.shuffle(patient_ids)
    num_val_patients = int(len(patient_ids) * val_fraction)
    num_test_patients = int(len(patient_ids) * test_fraction)

    test_patient_ids = patient_ids[:num_test_patients]
    val_patient_ids = patient_ids[num_test_patients:num_test_patients + num_val_patients]
    train_patient_ids = patient_ids[num_test_patients + num_val_patients:]

    train_indices = dataset.data[dataset.data['patient_id'].isin(train_patient_ids)].index.tolist()
    val_indices = dataset.data[dataset.data['patient_id'].isin(val_patient_ids)].index.tolist()
    test_indices = dataset.data[dataset.data['patient_id'].isin(test_patient_ids)].index.tolist()

    return train_indices, val_indices, test_indices
