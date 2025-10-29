"""
Data loader for IndicAccentDb dataset from HuggingFace.
Handles downloading, loading, and organizing the dataset.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import soundfile as sf
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class IndicAccentDataLoader:
    """
    Loader for the IndicAccentDb dataset.
    Dataset: https://huggingface.co/datasets/DarshanaS/IndicAccentDb
    """
    
    def __init__(
        self,
        dataset_name: str = "DarshanaS/IndicAccentDb",
        cache_dir: str = "./data/raw",
        processed_dir: str = "./data/processed",
        metadata_dir: str = "./data/metadata",
    ):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace
            cache_dir: Directory to cache downloaded data
            processed_dir: Directory to save processed data
            metadata_dir: Directory to save metadata
        """
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self.processed_dir = Path(processed_dir)
        self.metadata_dir = Path(metadata_dir)
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset = None
        self.label_mapping = {}
        self.reverse_label_mapping = {}
        
    def load_dataset(
        self,
        split: Optional[str] = None,
        streaming: bool = False
    ) -> DatasetDict:
        """
        Load the dataset from HuggingFace.
        
        Args:
            split: Specific split to load (train, test, etc.)
            streaming: Whether to use streaming mode
            
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        try:
            if split:
                self.dataset = load_dataset(
                    self.dataset_name,
                    split=split,
                    cache_dir=str(self.cache_dir),
                    streaming=streaming
                )
            else:
                self.dataset = load_dataset(
                    self.dataset_name,
                    cache_dir=str(self.cache_dir),
                    streaming=streaming
                )
            
            logger.info(f"Dataset loaded successfully: {self.dataset}")
            return self.dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def create_splits(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        seed: int = 42,
        stratify_by: str = "native_language"
    ) -> Dict[str, Dataset]:
        """
        Create train/val/test splits from the dataset.
        
        Args:
            train_size: Proportion of data for training
            val_size: Proportion of data for validation
            test_size: Proportion of data for testing
            seed: Random seed for reproducibility
            stratify_by: Column to stratify splits by
            
        Returns:
            Dictionary with train, val, and test datasets
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Split sizes must sum to 1.0"
        
        if self.dataset is None:
            self.load_dataset()
        
        # Convert to pandas for easier manipulation
        df = self.dataset.to_pandas() if hasattr(self.dataset, 'to_pandas') else pd.DataFrame(self.dataset)
        
        # Create label mapping
        if stratify_by in df.columns:
            unique_labels = sorted(df[stratify_by].unique())
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            self.reverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}
            
            # Add numeric labels
            df['label'] = df[stratify_by].map(self.label_mapping)
            stratify_col = df['label']
        else:
            stratify_col = None
        
        # First split: train and temp (val + test)
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            random_state=seed,
            stratify=stratify_col
        )
        
        # Second split: val and test
        val_ratio = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio,
            random_state=seed,
            stratify=temp_df['label'] if stratify_col is not None else None
        )
        
        # Convert back to Dataset objects
        splits = {
            'train': Dataset.from_pandas(train_df, preserve_index=False),
            'val': Dataset.from_pandas(val_df, preserve_index=False),
            'test': Dataset.from_pandas(test_df, preserve_index=False)
        }
        
        # Save metadata
        self._save_split_metadata(splits)
        self._save_label_mapping()
        
        logger.info(f"Created splits - Train: {len(splits['train'])}, "
                   f"Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        
        return splits
    
    def filter_by_age_group(
        self,
        dataset: Dataset,
        age_group: str
    ) -> Dataset:
        """
        Filter dataset by age group (adult or child).
        
        Args:
            dataset: Dataset to filter
            age_group: Age group to filter by ('adult' or 'child')
            
        Returns:
            Filtered dataset
        """
        if 'age_group' not in dataset.column_names:
            logger.warning("Age group column not found in dataset")
            return dataset
        
        filtered = dataset.filter(lambda x: x['age_group'] == age_group)
        logger.info(f"Filtered {age_group} samples: {len(filtered)}")
        return filtered
    
    def filter_by_speech_level(
        self,
        dataset: Dataset,
        speech_level: str
    ) -> Dataset:
        """
        Filter dataset by speech level (word or sentence).
        
        Args:
            dataset: Dataset to filter
            speech_level: Speech level to filter by ('word' or 'sentence')
            
        Returns:
            Filtered dataset
        """
        if 'speech_level' not in dataset.column_names:
            logger.warning("Speech level column not found in dataset")
            return dataset
        
        filtered = dataset.filter(lambda x: x['speech_level'] == speech_level)
        logger.info(f"Filtered {speech_level}-level samples: {len(filtered)}")
        return filtered
    
    def get_statistics(self, dataset: Optional[Dataset] = None) -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            dataset: Dataset to analyze (uses self.dataset if None)
            
        Returns:
            Dictionary with dataset statistics
        """
        if dataset is None:
            dataset = self.dataset
        
        if dataset is None:
            raise ValueError("No dataset loaded")
        
        df = dataset.to_pandas() if hasattr(dataset, 'to_pandas') else pd.DataFrame(dataset)
        
        stats = {
            'total_samples': len(df),
            'num_speakers': df['speaker_id'].nunique() if 'speaker_id' in df else None,
        }
        
        # Language distribution
        if 'native_language' in df.columns:
            stats['language_distribution'] = df['native_language'].value_counts().to_dict()
        
        # Age group distribution
        if 'age_group' in df.columns:
            stats['age_group_distribution'] = df['age_group'].value_counts().to_dict()
        
        # Speech level distribution
        if 'speech_level' in df.columns:
            stats['speech_level_distribution'] = df['speech_level'].value_counts().to_dict()
        
        # Audio duration statistics
        if 'duration' in df.columns:
            stats['duration_stats'] = {
                'mean': df['duration'].mean(),
                'std': df['duration'].std(),
                'min': df['duration'].min(),
                'max': df['duration'].max()
            }
        
        return stats
    
    def _save_split_metadata(self, splits: Dict[str, Dataset]):
        """Save metadata about the splits."""
        metadata = {}
        for split_name, split_data in splits.items():
            df = split_data.to_pandas() if hasattr(split_data, 'to_pandas') else pd.DataFrame(split_data)
            metadata[split_name] = {
                'num_samples': len(df),
                'language_distribution': df['native_language'].value_counts().to_dict() if 'native_language' in df else {}
            }
        
        metadata_path = self.metadata_dir / 'splits_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved split metadata to {metadata_path}")
    
    def _save_label_mapping(self):
        """Save label mapping to file."""
        import json
        mapping_path = self.metadata_dir / 'label_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(self.label_mapping, f, indent=2)
        
        logger.info(f"Saved label mapping to {mapping_path}")
    
    def load_label_mapping(self) -> Dict:
        """Load label mapping from file."""
        import json
        mapping_path = self.metadata_dir / 'label_mapping.json'
        
        if not mapping_path.exists():
            raise FileNotFoundError(f"Label mapping not found at {mapping_path}")
        
        with open(mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
        
        self.reverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}
        return self.label_mapping


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    loader = IndicAccentDataLoader()
    dataset = loader.load_dataset()
    
    # Create splits
    splits = loader.create_splits()
    
    # Get statistics
    stats = loader.get_statistics(splits['train'])
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
