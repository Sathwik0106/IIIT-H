"""
HuBERT feature extraction module.
Extracts self-supervised representations from HuBERT model.
"""

import logging
from typing import Optional, List, Dict, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import HubertModel, Wav2Vec2FeatureExtractor

logger = logging.getLogger(__name__)


class HuBERTFeatureExtractor:
    """
    Extracts features from HuBERT model with layer-wise analysis capability.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        extract_layers: Optional[List[int]] = None,
        pooling: str = "mean",
        normalize: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize HuBERT feature extractor.
        
        Args:
            model_name: HuBERT model name from HuggingFace
            extract_layers: List of layer indices to extract features from
            pooling: Pooling method ('mean', 'max', 'cls_token', or 'none')
            normalize: Whether to normalize features
            device: Device to run model on
            cache_dir: Cache directory for model weights
        """
        self.model_name = model_name
        self.extract_layers = extract_layers or [12]  # Default to last layer
        self.pooling = pooling
        self.normalize = normalize
        self.device = device
        
        logger.info(f"Loading HuBERT model: {model_name}")
        
        # Load feature extractor (processor)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Load HuBERT model
        self.model = HubertModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            output_hidden_states=True
        )
        self.model.to(device)
        self.model.eval()
        
        # Get model config
        self.sample_rate = self.processor.sampling_rate
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        
        logger.info(f"HuBERT model loaded: {self.num_layers} layers, "
                   f"hidden size: {self.hidden_size}")
        
        # Validate extract_layers
        for layer in self.extract_layers:
            if layer < 0 or layer > self.num_layers:
                raise ValueError(f"Invalid layer index: {layer}. "
                               f"Model has {self.num_layers} layers (0-{self.num_layers})")
    
    @torch.no_grad()
    def extract(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        layer: Optional[int] = None
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Extract features from audio.
        
        Args:
            audio: Audio signal (numpy array or torch tensor)
            layer: Specific layer to extract from (None = use self.extract_layers)
            
        Returns:
            Extracted features. If layer is specified, returns single array.
            Otherwise, returns dict mapping layer index to features.
        """
        # Convert to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Process audio through feature extractor
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.model(**inputs)
        
        # Extract hidden states
        # hidden_states: tuple of (num_layers + 1) tensors
        # Each tensor shape: (batch_size, sequence_length, hidden_size)
        hidden_states = outputs.hidden_states
        
        # Determine which layers to extract
        layers_to_extract = [layer] if layer is not None else self.extract_layers
        
        # Extract and pool features
        features = {}
        for layer_idx in layers_to_extract:
            # Get layer output (layer 0 is input, layer 1 is first transformer layer)
            layer_output = hidden_states[layer_idx]
            
            # Apply pooling
            pooled = self._pool_features(layer_output)
            
            # Normalize if requested
            if self.normalize:
                pooled = self._normalize(pooled)
            
            # Convert to numpy
            features[layer_idx] = pooled.cpu().numpy().squeeze()
        
        # Return single array if single layer, otherwise dict
        if layer is not None:
            return features[layer]
        elif len(features) == 1:
            return list(features.values())[0]
        else:
            return features
    
    def extract_all_layers(
        self,
        audio: Union[np.ndarray, torch.Tensor]
    ) -> Dict[int, np.ndarray]:
        """
        Extract features from all layers.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary mapping layer index to features
        """
        # Temporarily set extract_layers to all layers
        original_layers = self.extract_layers
        self.extract_layers = list(range(self.num_layers + 1))
        
        features = self.extract(audio)
        
        # Restore original layers
        self.extract_layers = original_layers
        
        return features
    
    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling to sequence features.
        
        Args:
            features: Feature tensor (batch_size, sequence_length, hidden_size)
            
        Returns:
            Pooled features (batch_size, hidden_size)
        """
        if self.pooling == "mean":
            # Mean pooling across time
            return torch.mean(features, dim=1)
        
        elif self.pooling == "max":
            # Max pooling across time
            return torch.max(features, dim=1)[0]
        
        elif self.pooling == "cls_token":
            # Use first token (CLS token)
            return features[:, 0, :]
        
        elif self.pooling == "none":
            # No pooling, return full sequence
            return features
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features using L2 normalization.
        
        Args:
            features: Feature tensor
            
        Returns:
            Normalized features
        """
        return torch.nn.functional.normalize(features, p=2, dim=-1)
    
    def get_embedding_dim(self, layer: Optional[int] = None) -> int:
        """
        Get the dimensionality of extracted features.
        
        Args:
            layer: Layer index (unused, kept for API compatibility)
            
        Returns:
            Feature dimensionality
        """
        if self.pooling == "none":
            raise ValueError("Cannot determine fixed dimension for unpooled features")
        return self.hidden_size
    
    def __call__(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        layer: Optional[int] = None
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Callable interface for feature extraction.
        
        Args:
            audio: Audio signal
            layer: Specific layer to extract from
            
        Returns:
            Extracted features
        """
        return self.extract(audio, layer)


class HuBERTLayerAnalyzer:
    """
    Analyzes which HuBERT layer best captures accent information.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize layer analyzer.
        
        Args:
            model_name: HuBERT model name
            device: Device to run on
        """
        self.feature_extractor = HuBERTFeatureExtractor(
            model_name=model_name,
            device=device
        )
        self.layer_features = {}
    
    def extract_layer_features(
        self,
        audio_samples: List[np.ndarray],
        labels: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Extract features from all layers for multiple audio samples.
        
        Args:
            audio_samples: List of audio arrays
            labels: Optional list of labels
            
        Returns:
            Dictionary mapping layer index to feature matrix
        """
        logger.info(f"Extracting features from {len(audio_samples)} samples")
        
        # Extract features for each sample
        all_layer_features = {i: [] for i in range(self.feature_extractor.num_layers + 1)}
        
        for audio in audio_samples:
            layer_features = self.feature_extractor.extract_all_layers(audio)
            
            for layer_idx, features in layer_features.items():
                all_layer_features[layer_idx].append(features)
        
        # Stack features
        for layer_idx in all_layer_features:
            all_layer_features[layer_idx] = np.stack(all_layer_features[layer_idx])
        
        self.layer_features = all_layer_features
        
        logger.info(f"Extracted features from {len(all_layer_features)} layers")
        return all_layer_features
    
    def compute_layer_separability(
        self,
        labels: np.ndarray
    ) -> Dict[int, float]:
        """
        Compute class separability for each layer using Fisher criterion.
        
        Args:
            labels: Array of class labels
            
        Returns:
            Dictionary mapping layer index to separability score
        """
        if not self.layer_features:
            raise ValueError("No layer features extracted. Call extract_layer_features first.")
        
        separability_scores = {}
        
        for layer_idx, features in self.layer_features.items():
            # Compute between-class and within-class scatter
            score = self._compute_fisher_criterion(features, labels)
            separability_scores[layer_idx] = score
        
        return separability_scores
    
    def _compute_fisher_criterion(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute Fisher criterion for class separability.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Label array (n_samples,)
            
        Returns:
            Fisher criterion score
        """
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        if n_classes < 2:
            return 0.0
        
        # Overall mean
        overall_mean = np.mean(features, axis=0)
        
        # Between-class scatter
        Sb = np.zeros((features.shape[1], features.shape[1]))
        # Within-class scatter
        Sw = np.zeros((features.shape[1], features.shape[1]))
        
        for label in unique_labels:
            class_features = features[labels == label]
            class_mean = np.mean(class_features, axis=0)
            n_samples = len(class_features)
            
            # Between-class
            mean_diff = (class_mean - overall_mean).reshape(-1, 1)
            Sb += n_samples * (mean_diff @ mean_diff.T)
            
            # Within-class
            for sample in class_features:
                sample_diff = (sample - class_mean).reshape(-1, 1)
                Sw += sample_diff @ sample_diff.T
        
        # Compute Fisher criterion: trace(Sb) / trace(Sw)
        fisher_score = np.trace(Sb) / (np.trace(Sw) + 1e-10)
        
        return fisher_score


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy audio
    sample_rate = 16000
    duration = 3  # seconds
    audio = np.random.randn(sample_rate * duration)
    
    # Extract features
    extractor = HuBERTFeatureExtractor(
        model_name="facebook/hubert-base-ls960",
        extract_layers=[6, 12],
        device="cpu"  # Use CPU for testing
    )
    
    features = extractor.extract(audio)
    
    if isinstance(features, dict):
        for layer, feat in features.items():
            print(f"Layer {layer} features shape: {feat.shape}")
    else:
        print(f"Features shape: {features.shape}")
