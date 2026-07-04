"""
PyTorch Dataset Pipeline for ASL Recognition
- Loads metadata from the generated CSV.
- Dynamically loads (30, 225) .npy feature tensors.
- Provides DataLoaders for train, validation, and test subsets.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict

class ASLDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MediaPipe extracted ASL features.
    """
    def __init__(self, metadata_csv: str, subset: str = "train"):
        """
        Args:
            metadata_csv (str): Path to the metadata_mp.csv file.
            subset (str): "train", "val", or "test".
        """
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata CSV not found at: {metadata_csv}")
        
        # Load the CSV using pandas
        self.metadata = pd.read_csv(metadata_csv)
        
        # Filter the dataset by the requested subset
        self.metadata = self.metadata[self.metadata["subset"] == subset].reset_index(drop=True)
        
        if len(self.metadata) == 0:
            raise ValueError(f"No records found for subset: '{subset}'. Check your CSV.")

    def __len__(self) -> int:
        """Returns the total number of videos in this subset."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads the .npy tensor and its corresponding action label.
        
        Args:
            idx (int): Row index in the filtered dataframe.
            
        Returns:
            features (torch.Tensor): The (30, 225) float32 tensor.
            label (torch.Tensor): The integer class label.
        """
        row = self.metadata.iloc[idx]
        feature_path = row["feature_path"]
        action_label = row["action_label"]

        # Load the numpy array from disk
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature tensor not found: {feature_path}")

        features_np = np.load(feature_path)
        
        # Convert to PyTorch Tensors
        # Features must be float32 for model weights, labels must be long for CrossEntropyLoss
        features_tensor = torch.tensor(features_np, dtype=torch.float32)
        label_tensor = torch.tensor(action_label, dtype=torch.long)

        return features_tensor, label_tensor


def get_dataloaders(
    metadata_csv: str, 
    batch_size: int = 32, 
    num_workers: int = 2
) -> Dict[str, DataLoader]:
    """
    Factory function to create DataLoaders for all three subsets.
    
    Args:
        metadata_csv (str): Path to metadata_mp.csv.
        batch_size (int): Number of videos per batch.
        num_workers (int): Number of CPU subprocesses for data loading.
        
    Returns:
        Dict containing the 'train', 'val', and 'test' DataLoaders.
    """
    dataloaders = {}
    
    for subset in ["train", "val", "test"]:
        dataset = ASLDataset(metadata_csv=metadata_csv, subset=subset)
        
        # Shuffle only the training data to prevent the model from learning the sequence order
        shuffle = True if subset == "train" else False
        
        dataloaders[subset] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=True # Speeds up transfer from CPU to GPU
        )
        
    return dataloaders


# ============================================================
# TESTING ROUTINE
# ============================================================
if __name__ == "__main__":
    # Test the dataset logic directly if this script is run
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ASL Dataset loading")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to metadata_mp.csv")
    args = parser.parse_args()
    
    try:
        loaders = get_dataloaders(args.csv_path, batch_size=4)
        
        print("\n✅ DataLoaders initialized successfully.")
        
        for subset, loader in loaders.items():
            print(f"Subset: {subset.upper()} | Total Batches: {len(loader)} | Total Samples: {len(loader.dataset)}")
            
            # Fetch a single batch to test dimensions
            features, labels = next(iter(loader))
            print(f"  -> Batch Features Shape: {features.shape} # Expected: (Batch, 30, 225)")
            print(f"  -> Batch Labels Shape: {labels.shape} # Expected: (Batch)")
            break # Only test the first loader to confirm it works
            
    except Exception as e:
        print(f"\n❌ Error initializing dataset: {e}")