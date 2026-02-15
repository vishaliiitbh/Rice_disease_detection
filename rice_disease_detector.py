"""
Rice Leaf Disease Detection Model
Auto-generated model wrapper for easy deployment
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class BaselineModel(nn.Module):
    """MobileNetV2-based model for rice leaf disease classification"""

    def __init__(self, num_classes, pretrained=False):
        super(BaselineModel, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class RiceLeafDiseaseDetector:
    """
    Complete Rice Leaf Disease Detection Model

    Usage:
        model = RiceLeafDiseaseDetector.load('rice_disease_model.pth')
        result = model.predict('path/to/image.jpg')
    """

    def __init__(self, model, classes, image_size=224, device='cpu'):
        self.model = model
        self.classes = classes
        self.image_size = image_size
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, return_probs=False):
        """
        Predict disease class for a rice leaf image

        Args:
            image_path: Path to image file (str) or PIL Image object
            return_probs: If True, returns all class probabilities

        Returns:
            Dictionary with predicted_class, confidence, and optionally all_probabilities
        """
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = self.classes[predicted_idx.item()]
        confidence_score = float(confidence.item())

        result = {
            'predicted_class': predicted_class,
            'confidence': confidence_score
        }

        if return_probs:
            all_probs = probabilities.cpu().numpy()[0]
            prob_dict = {self.classes[i]: float(all_probs[i]) for i in range(len(self.classes))}
            result['all_probabilities'] = prob_dict

        return result

    def predict_batch(self, image_paths, return_probs=False):
        """Predict on multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path, return_probs=return_probs)
            result['image_path'] = img_path
            results.append(result)
        return results

    def save(self, filepath):
        """Save the complete model"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'classes': self.classes,
            'image_size': self.image_size,
            'model_architecture': 'MobileNetV2'
        }
        torch.save(save_dict, filepath)
        print(f"✓ Model saved to {filepath}")

    @classmethod
    def load(cls, filepath, device='cpu'):
        """Load a saved model"""
        checkpoint = torch.load(filepath, map_location=device)

        num_classes = len(checkpoint['classes'])
        model = BaselineModel(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        detector = cls(
            model=model,
            classes=checkpoint['classes'],
            image_size=checkpoint['image_size'],
            device=device
        )

        print(f"✓ Model loaded from {filepath}")
        print(f"  Classes: {len(detector.classes)}")
        print(f"  Device: {device}")

        return detector

    def get_classes(self):
        """Get list of disease classes"""
        return self.classes

    def __repr__(self):
        return f"RiceLeafDiseaseDetector(classes={len(self.classes)}, device={self.device})"


# Convenience function for quick predictions
def quick_predict(model_path, image_path):
    """
    Quick prediction function

    Args:
        model_path: Path to saved model
        image_path: Path to image

    Returns:
        Prediction result dictionary
    """
    model = RiceLeafDiseaseDetector.load(model_path)
    return model.predict(image_path, return_probs=True)