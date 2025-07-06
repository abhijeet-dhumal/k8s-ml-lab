#!/usr/bin/env python3
"""
MNIST Model Inference Script

Test the trained MNIST CNN model by providing handwritten digit images
and getting predicted digit outputs.

Usage:
    python test_mnist_model.py --image path/to/image.png
    python test_mnist_model.py --image path/to/image.png --model path/to/model.pth
    python test_mnist_model.py --batch path/to/images/directory/
    python test_mnist_model.py --interactive  # Draw digits in terminal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os
import sys
import glob
from pathlib import Path

# Import the same CNN model architecture used for training
class CNNModel(nn.Module):
    """Same CNN model architecture as used in training"""
    
    def __init__(self):
        super(CNNModel, self).__init__()
        # Balanced model for good accuracy with reasonable resources
        self.conv1 = nn.Conv2d(1, 16, 5, 1)  # 16 channels
        self.conv2 = nn.Conv2d(16, 32, 5, 1)  # 32 channels
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(512, 64)  # 64 neurons
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

class MNISTPredictor:
    """MNIST digit predictor using trained CNN model"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cpu')  # Use CPU for inference
        self.model = CNNModel()
        
        # Find model file if not specified
        if model_path is None:
            model_path = self.find_model_file()
        
        # Load trained model
        self.load_model(model_path)
        
        # Image preprocessing pipeline (same as training)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.Resize((28, 28)),                  # Resize to MNIST size
            transforms.ToTensor(),                        # Convert to tensor
            transforms.Normalize((0.1307,), (0.3081,))   # MNIST normalization
        ])
    
    def find_model_file(self):
        """Find the most recent trained model file"""
        possible_paths = [
            'output/latest/trained-model.pth',
            'output/trained-model.pth',
            '../output/latest/trained-model.pth',
            '../output/trained-model.pth'
        ]
        
        # Also search for any .pth files in output directories
        search_patterns = [
            'output/**/trained-model.pth',
            '../output/**/trained-model.pth',
            'output/**/*.pth',
            '../output/**/*.pth'
        ]
        
        # Try exact paths first
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found model: {path}")
                return path
        
        # Search with patterns
        for pattern in search_patterns:
            files = glob.glob(pattern, recursive=True)
            if files:
                # Use the most recent file
                latest_file = max(files, key=os.path.getmtime)
                print(f"Found model: {latest_file}")
                return latest_file
        
        raise FileNotFoundError(
            "No trained model found. Please run training first or specify --model path"
        )
    
    def load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load model state
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Set to evaluation mode
            print(f"✅ Model loaded successfully from: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def preprocess_image(self, image_path):
        """Preprocess input image for model inference"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path)
            else:
                image = image_path  # Already a PIL Image
            
            # Convert to RGB first (handles RGBA, etc.)
            image = image.convert('RGB')
            
            # Apply preprocessing pipeline
            tensor = self.transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess image: {e}")
    
    def predict(self, image_input, show_confidence=True):
        """Predict digit from image"""
        # Preprocess image
        if isinstance(image_input, str):
            tensor = self.preprocess_image(image_input)
            image_name = os.path.basename(image_input)
        else:
            tensor = self.preprocess_image(image_input)
            image_name = "input_image"
        
        # Run inference
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_digit = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_digit].item()
        
        result = {
            'image': image_name,
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities[0].tolist()
        }
        
        if show_confidence:
            print(f"📸 Image: {image_name}")
            print(f"🔢 Predicted digit: {predicted_digit}")
            print(f"📊 Confidence: {confidence:.2%}")
            
            # Show top 3 predictions
            top_3_indices = probabilities[0].argsort(descending=True)[:3]
            print("🏆 Top 3 predictions:")
            for i, idx in enumerate(top_3_indices):
                digit = idx.item()
                prob = probabilities[0][idx].item()
                print(f"   {i+1}. Digit {digit}: {prob:.2%}")
            print()
        
        return result
    
    def predict_batch(self, image_directory, pattern="*.png"):
        """Predict digits for all images in a directory"""
        image_dir = Path(image_directory)
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_directory}")
        
        # Find all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
            image_files.extend(glob.glob(str(image_dir / ext)))
        
        if not image_files:
            print(f"No image files found in {image_directory}")
            return []
        
        print(f"Found {len(image_files)} images in {image_directory}")
        print("="*50)
        
        results = []
        for image_file in sorted(image_files):
            try:
                result = self.predict(image_file)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {image_file}: {e}")
        
        # Summary
        print("="*50)
        print(f"📊 Batch Prediction Summary:")
        print(f"   Total images: {len(image_files)}")
        print(f"   Successful predictions: {len(results)}")
        
        if results:
            avg_confidence = np.mean([r['confidence'] for r in results])
            print(f"   Average confidence: {avg_confidence:.2%}")
        
        return results



def main():
    parser = argparse.ArgumentParser(description='Test trained MNIST CNN model')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, help='Path to trained model (.pth file)')
    parser.add_argument('--batch', type=str, help='Directory containing images for batch prediction')

    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.image, args.batch]):
        print("❌ Error: Please specify either --image or --batch")
        print("Please provide your own handwritten digit images for testing")
        parser.print_help()
        return
    
    try:
        # Initialize predictor
        print("🤖 Initializing MNIST predictor...")
        predictor = MNISTPredictor(args.model)
        print()
        
        # Single image prediction
        if args.image:
            if not os.path.exists(args.image):
                print(f"❌ Image file not found: {args.image}")
                return
            
            print("🔍 Running single image prediction...")
            result = predictor.predict(args.image)
            
        # Batch prediction
        elif args.batch:
            print("🔍 Running batch prediction...")
            results = predictor.predict_batch(args.batch)
            
        print("✅ Prediction completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 