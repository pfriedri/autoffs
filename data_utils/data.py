import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from data_utils.data_utils import load_and_process_data

class SkullDataset(Dataset):
    def __init__(self, data_dir, metadata_csv, augment=False, front_only=False):
        self.data_dir = data_dir
        self.augment = augment
        self.front_only=front_only

        # Load metadata. The SMSC metadata file uses 'Gender' for the M/F
        # column; older CSVs use 'sex'. Alias to a single name so the rest of
        # the code (and the 'sex' key in __getitem__) is agnostic.
        df = pd.read_csv(metadata_csv)
        if 'Gender' in df.columns and 'sex' not in df.columns:
            df = df.rename(columns={'Gender': 'sex'})
        self.data = df[df['kept'] == True].reset_index(drop=True)
        self.data = self._add_modality()

    def _add_modality(self):
        data = self.data.copy()
        data['modality'] = 'segmentation.nii.gz'
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir,
                            str(self.data.iloc[idx]['patient_id']),
                            str(self.data.iloc[idx]['visit_id']),
                            self.data.iloc[idx]['modality'])
        img = load_and_process_data(path, augment=self.augment)

        # Use bone only (labels 7 & 8) -> binarize
        img = np.where((img == 7) | (img == 8), 1, 0)

        # Only use frontal half of the skull
        if self.front_only:
            img = img[:, :, img.shape[2]//2:, :]

        patient_id = self.data.iloc[idx]['patient_id']
        visit_id = self.data.iloc[idx]['visit_id']
        sex = self.data.iloc[idx]['sex']
        sex = 0 if sex == 'M' else 1  # Encode Sex M -> 0, F -> 1

        sample = {
            'image': torch.tensor(img, dtype=torch.float32),
            'patient_id': patient_id,
            'visit_id': visit_id,
            'sex': sex
        }
        return sample

    def get_patient_ids(self):
        return self.data['patient_id'].unique()
