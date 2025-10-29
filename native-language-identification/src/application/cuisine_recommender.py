"""
Accent-Aware Cuisine Recommendation Application.
Uses trained NLI model to recommend regional cuisines based on detected accent.
"""

import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CuisineRecommender:
    """
    Recommends cuisines based on detected native language/accent.
    """
    
    def __init__(
        self,
        model: nn.Module,
        label_mapping: Dict[str, int],
        cuisine_mapping: Dict[str, List[str]],
        preprocessor: any,
        feature_extractor: any,
        device: str = "cuda",
        confidence_threshold: float = 0.6
    ):
        """
        Initialize cuisine recommender.
        
        Args:
            model: Trained NLI model
            label_mapping: Mapping from language names to indices
            cuisine_mapping: Mapping from languages to cuisine lists
            preprocessor: Audio preprocessor
            feature_extractor: Feature extractor
            device: Device to run on
            confidence_threshold: Minimum confidence for recommendations
        """
        self.model = model.to(device)
        self.model.eval()
        
        self.label_mapping = label_mapping
        self.reverse_label_mapping = {v: k for k, v in label_mapping.items()}
        self.cuisine_mapping = cuisine_mapping
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        logger.info("Initialized Cuisine Recommender")
    
    def predict_language(
        self,
        audio: np.ndarray
    ) -> Tuple[str, float, np.ndarray]:
        """
        Predict native language from audio.
        
        Args:
            audio: Audio array
            
        Returns:
            Tuple of (predicted language, confidence, all probabilities)
        """
        # Preprocess audio
        processed_audio = self.preprocessor.preprocess(audio)
        
        # Extract features
        features = self.feature_extractor(processed_audio)
        
        # Convert to tensor
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        # Add batch dimension
        features = features.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get language name
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        language = self.reverse_label_mapping.get(predicted_idx, "Unknown")
        
        return language, confidence, probabilities.cpu().numpy()[0]
    
    def recommend_cuisines(
        self,
        audio: np.ndarray,
        top_k: int = 3
    ) -> Dict:
        """
        Recommend cuisines based on audio.
        
        Args:
            audio: Audio array
            top_k: Number of cuisines to recommend
            
        Returns:
            Dictionary with predictions and recommendations
        """
        # Predict language
        language, confidence, probabilities = self.predict_language(audio)
        
        # Get cuisines
        cuisines = self.cuisine_mapping.get(language, ["No specific recommendations"])
        
        # Select top-k cuisines
        recommended_cuisines = cuisines[:top_k] if len(cuisines) >= top_k else cuisines
        
        # Prepare response
        response = {
            'detected_language': language,
            'confidence': confidence,
            'confidence_percentage': confidence * 100,
            'all_probabilities': {
                self.reverse_label_mapping[i]: prob 
                for i, prob in enumerate(probabilities)
            },
            'recommended_cuisines': recommended_cuisines,
            'reliable': confidence >= self.confidence_threshold
        }
        
        return response
    
    def generate_recommendation_message(
        self,
        audio: np.ndarray,
        top_k: int = 3
    ) -> str:
        """
        Generate human-readable recommendation message.
        
        Args:
            audio: Audio array
            top_k: Number of cuisines to recommend
            
        Returns:
            Recommendation message string
        """
        result = self.recommend_cuisines(audio, top_k)
        
        language = result['detected_language']
        confidence = result['confidence_percentage']
        cuisines = result['recommended_cuisines']
        reliable = result['reliable']
        
        if not reliable:
            message = (
                f"I detected a {language} accent (confidence: {confidence:.1f}%), "
                f"but I'm not very confident. Here are some general recommendations: "
                f"{', '.join(cuisines[:top_k])}"
            )
        else:
            message = (
                f"Based on your {language} accent (confidence: {confidence:.1f}%), "
                f"I recommend trying these delicious dishes: {', '.join(cuisines[:top_k])}. "
                f"Would you like to know more about any of these?"
            )
        
        return message


class RestaurantApplication:
    """
    Complete restaurant application with accent detection.
    """
    
    def __init__(
        self,
        recommender: CuisineRecommender
    ):
        """
        Initialize restaurant application.
        
        Args:
            recommender: Cuisine recommender instance
        """
        self.recommender = recommender
        logger.info("Restaurant application initialized")
    
    def process_customer_speech(
        self,
        audio: np.ndarray,
        customer_name: Optional[str] = None
    ) -> str:
        """
        Process customer speech and provide recommendations.
        
        Args:
            audio: Customer audio
            customer_name: Optional customer name
            
        Returns:
            Personalized recommendation message
        """
        # Get recommendations
        result = self.recommender.recommend_cuisines(audio)
        
        # Generate greeting
        greeting = f"Welcome, {customer_name}! " if customer_name else "Welcome! "
        
        # Generate message
        message = greeting + self.recommender.generate_recommendation_message(audio)
        
        logger.info(f"Processed customer speech: {result['detected_language']} "
                   f"({result['confidence_percentage']:.1f}%)")
        
        return message
    
    def display_menu_suggestions(
        self,
        audio: np.ndarray
    ) -> Dict:
        """
        Display personalized menu suggestions.
        
        Args:
            audio: Customer audio
            
        Returns:
            Menu suggestions with details
        """
        result = self.recommender.recommend_cuisines(audio, top_k=5)
        
        menu_suggestions = {
            'detected_accent': f"{result['detected_language']} ({result['confidence_percentage']:.1f}%)",
            'recommended_items': result['recommended_cuisines'],
            'personalization_note': (
                f"These recommendations are based on your {result['detected_language']} accent. "
                f"We specialize in authentic regional cuisines!"
            )
        }
        
        return menu_suggestions


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would normally load a trained model
    # For demo purposes, we'll show the structure
    
    print("Accent-Aware Cuisine Recommendation System")
    print("=" * 50)
    print("\nExample workflow:")
    print("1. Customer speaks a short English phrase")
    print("2. System detects Malayalam accent")
    print("3. System recommends: Appam, Puttu, Avial")
    print("\nThis creates a personalized dining experience!")
