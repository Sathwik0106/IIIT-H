"""
PyTorch Dataset class for Native Language Identification.
"""

import logging
from typing import Dict, Optional, Callable, Union, List
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset

from .preprocessing import AudioPreprocessor, AudioAugmentor

logger = logging.getLogger(__name__)


class NLIDataset(Dataset):
    """
    PyTorch Dataset for Native Language Identification.
    """
    
    def __init__(
        self,
        data: Union[HFDataset, List[Dict]],
        preprocessor: AudioPreprocessor,
        feature_extractor: Optional[Callable] = None,
        augmentor: Optional[AudioAugmentor] = None,
        label_mapping: Optional[Dict] = None,
        audio_column: str = "audio",
        label_column: str = "native_language",
        apply_augmentation: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            data: HuggingFace dataset or list of data dictionaries
            preprocessor: Audio preprocessor
            feature_extractor: Optional feature extractor function
            augmentor: Optional audio augmentor
            label_mapping: Mapping from language names to indices
            audio_column: Name of audio column in data
            label_column: Name of label column in data
            apply_augmentation: Whether to apply augmentation
        """
        self.data = data
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.augmentor = augmentor
        self.label_mapping = label_mapping
        self.audio_column = audio_column
        self.label_column = label_column
        self.apply_augmentation = apply_augmentation
        
        # Create label mapping if not provided
        if self.label_mapping is None and hasattr(data, '__len__'):
            unique_labels = set()
            for item in data:
                if label_column in item:
                    unique_labels.add(item[label_column])
            self.label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        
        logger.info(f"Initialized dataset with {len(self)} samples")
        if self.label_mapping:
            logger.info(f"Label mapping: {self.label_mapping}")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with audio tensor and label
        """
        # Get data item
        item = self.data[idx]
        
        # Extract audio
        if self.audio_column in item:
            audio_data = item[self.audio_column]
            
            # Handle different audio formats
            if isinstance(audio_data, dict):
                # HuggingFace audio format
                audio = np.array(audio_data['array'])
            elif isinstance(audio_data, (str, Path)):
                # File path
                audio, _ = self.preprocessor.load_audio(audio_data)
            else:
                # Assume it's already a numpy array
                audio = np.array(audio_data)
        else:
            raise ValueError(f"Audio column '{self.audio_column}' not found in data")
        
        # Preprocess audio
        audio = self.preprocessor.preprocess(audio)
        
        # Apply augmentation if enabled
        if self.apply_augmentation and self.augmentor is not None:
            audio = self.augmentor.augment(audio)
        
        # Extract features if feature extractor is provided
        if self.feature_extractor is not None:
            features = self.feature_extractor(audio)
        else:
            features = audio
        
        # Convert to tensor
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        elif not isinstance(features, torch.Tensor):
            features = torch.FloatTensor([features])
        
        # Get label
        if self.label_column in item:
            label_name = item[self.label_column]
            if self.label_mapping:
                label = self.label_mapping.get(label_name, -1)
            else:
                label = label_name
        else:
            label = -1
        
        label_tensor = torch.LongTensor([label])
        
        # Prepare output
        output = {
            'features': features,
            'label': label_tensor,
            'audio': torch.FloatTensor(audio)
        }
        
        # Add metadata if available
        if 'speaker_id' in item:
            output['speaker_id'] = item['speaker_id']
        if 'age_group' in item:
            output['age_group'] = item['age_group']
        if 'speech_level' in item:
            output['speech_level'] = item['speech_level']
        
        return output


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors
    """
    # Stack features
    features = torch.stack([item['features'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])
    audios = torch.stack([item['audio'] for item in batch])
    
    output = {
        'features': features,
        'label': labels,
        'audio': audios
    }
    
    # Add metadata if available
    if 'speaker_id' in batch[0]:
        output['speaker_id'] = [item['speaker_id'] for item in batch]
    if 'age_group' in batch[0]:
        output['age_group'] = [item['age_group'] for item in batch]
    if 'speech_level' in batch[0]:
        output['speech_level'] = [item['speech_level'] for item in batch]
    
    return output


def create_dataloader(
    dataset: NLIDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: NLI dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    dummy_data = [
        {
            'audio': {'array': np.random.randn(16000 * 3)},
            'native_language': 'Hindi',
            'speaker_id': 'speaker_001'
        },
        {
            'audio': {'array': np.random.randn(16000 * 2)},
            'native_language': 'Tamil',
            'speaker_id': 'speaker_002'
        }
    ]
    
    # Create preprocessor
    preprocessor = AudioPreprocessor(sample_rate=16000)
    
    # Create dataset
    dataset = NLIDataset(
        data=dummy_data,
        preprocessor=preprocessor
    )
    
    # Create dataloader
    dataloader = create_dataloader(dataset, batch_size=2, num_workers=0)
    
    # Test
    for batch in dataloader:
        print(f"Features shape: {batch['features'].shape}")
        print(f"Labels shape: {batch['label'].shape}")
        break
