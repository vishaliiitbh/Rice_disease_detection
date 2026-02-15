<div align="center">

# 🌾 Rice Leaf Disease Detection
### Edge AI Solution for Agricultural Disease Classification

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai/)
[![Flutter](https://img.shields.io/badge/Flutter-3.0+-02569B?style=for-the-badge&logo=flutter&logoColor=white)](https://flutter.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Documentation](#-documentation)

---

### 🎯 An end-to-end Edge AI solution for rice leaf disease detection, optimized for mobile deployment with **74.5% size reduction** and **only 0.7% accuracy loss**.

</div>

---

## 📸 Demo

<div align="center">

| Mobile App | Inference Results | Probability Distribution |
|:----------:|:-----------------:|:------------------------:|
| ![Home Screen](assets/home.png) | ![Result](assets/result.png) | ![Chart](assets/chart.png) |

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Trade-off Analysis](#-trade-off-analysis)
- [Installation](#-installation)
- [Usage](#-usage)
- [Mobile App](#-mobile-app)
- [Results](#-results)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a **production-ready** rice leaf disease classification system optimized for edge deployment. The system classifies rice leaves into 6 categories with high accuracy while maintaining efficiency for mobile devices.

### 🌟 Disease Categories

<div align="center">

| Disease | Description |
|:--------|:------------|
| 🦠 **Bacterial Leaf Blight** | Common bacterial infection |
| 🟤 **Brown Spot** | Fungal disease affecting leaves |
| ✅ **Healthy Rice Leaf** | No disease detected |
| 💥 **Leaf Blast** | Serious fungal disease |
| 🌊 **Leaf Scald** | Water-borne pathogen |
| 🛡️ **Sheath Blight** | Soil-borne fungal disease |

</div>

### 🎨 Key Highlights

```diff
+ 89.20% baseline accuracy on test set
+ 88.50% edge model accuracy (only 0.7% drop)
+ 74.5% model size reduction (8.75 MB → 2.23 MB)
+ 71.6% faster inference (45.32 ms → 12.85 ms)
+ 3.5x speed improvement on CPU
+ Full offline capability for mobile deployment
```

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🚀 Performance
- **High Accuracy**: 89.20% on baseline, 88.50% on edge
- **Fast Inference**: 12.85ms average on CPU
- **Small Model**: Only 2.23 MB for mobile deployment
- **Efficient**: 3.5x faster than baseline

</td>
<td width="50%">

### 🛠️ Technical
- **Transfer Learning**: MobileNetV2 architecture
- **Optimization**: INT8 dynamic quantization
- **Cross-Platform**: ONNX Runtime support
- **Mobile-Ready**: Flutter app included

</td>
</tr>
<tr>
<td width="50%">

### 📱 Deployment
- **Offline Capable**: No internet required
- **Real-time**: Sub-100ms inference
- **User-Friendly**: Simple camera interface
- **Production-Ready**: Complete mobile app

</td>
<td width="50%">

### 📚 Documentation
- **Comprehensive**: Full technical approach
- **Reproducible**: Complete training pipeline
- **Well-Tested**: Evaluation scripts included
- **Open Source**: MIT licensed

</td>
</tr>
</table>

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/rice-disease-detection.git
cd rice-disease-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run inference on a test image
python scripts/predict.py \
    --model models/edge_model.onnx \
    --image path/to/rice_leaf.jpg \
    --visualize

# 4. (Optional) Run the mobile app
cd mobile_app
flutter pub get
flutter run --release
```

**That's it!** 🎉 You should see the disease prediction with confidence score.

---

## 📊 Trade-off Analysis

<div align="center">

### **Baseline vs Edge Model Comparison**

| Metric | Baseline Model | Edge Model | Change |
|:-------|:--------------:|:----------:|:------:|
| **📊 Accuracy** | 89.20% | 88.50% | <span style="color: orange;">↓ 0.7%</span> |
| **💾 File Size** | 8.75 MB | 2.23 MB | <span style="color: green;">↓ 74.5%</span> |
| **⚡ Inference Speed** | 45.32 ms | 12.85 ms | <span style="color: green;">↓ 71.6%</span> |
| **🎯 Format** | PyTorch (.pth) | ONNX (.onnx) | Optimized |
| **🔢 Precision** | FP32 | INT8 | 4x smaller |
| **📱 Mobile Ready** | ❌ | ✅ | Yes |

</div>

### 💡 Why This Trade-off is Excellent

For **only 0.7% accuracy loss**, we gained:

- ✅ **3.5x faster inference** - Better user experience
- ✅ **74.5% smaller model** - Fits easily in mobile apps
- ✅ **Lower power consumption** - Extended battery life
- ✅ **Offline capability** - Works anywhere, anytime
- ✅ **Privacy preserved** - Data never leaves device
- ✅ **No server costs** - Complete edge deployment

<details>
<summary><b>📈 Hardware Specifications</b></summary>

**Inference Benchmarking:**
- **CPU**: Intel Core i7-9750H @ 2.60GHz
- **RAM**: 16 GB DDR4
- **OS**: Ubuntu 24.04 LTS
- **Mode**: CPU only (no GPU acceleration)

**Training:**
- **GPU**: NVIDIA Tesla T4
- **Training Time**: ~45 minutes

</details>

---

## 📊 Dataset

**Source**: Rice Leaf Disease Dataset (Kaggle)

**Statistics**:
```
Total Images:    3,829
Classes:         6
Train Split:     70% (2,680 images)
Validation:      15% (574 images)
Test Split:      15% (575 images)
Image Size:      224x224 pixels
```

**Class Distribution**:
```
✓ Bacterial Leaf Blight:  636 images (16.6%)
✓ Brown Spot:             646 images (16.9%)
✓ Healthy Rice Leaf:      653 images (17.1%)
✓ Leaf Blast:             634 images (16.6%)
✓ Leaf Scald:             628 images (16.4%)
✓ Sheath Blight:          632 images (16.5%)
```

**Note**: Well-balanced dataset with <1% variation between classes.

---

## 🏗️ Model Architecture

### Baseline Model

**Architecture**: MobileNetV2 (Transfer Learning from ImageNet)

```
┌─────────────────────────────────────────────────────┐
│              MobileNetV2 Architecture                │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Input (224×224×3)                                  │
│         ↓                                            │
│  Conv2d (3→32, stride=2)                            │
│         ↓                                            │
│  Inverted Residual Blocks (×17)                     │
│  ├─ Expansion (1×1 conv)                            │
│  ├─ Depthwise (3×3 conv)                            │
│  ├─ Projection (1×1 conv)                           │
│  └─ Skip connection                                 │
│         ↓                                            │
│  Conv2d (320→1280)                                  │
│         ↓                                            │
│  Global Average Pooling                             │
│         ↓                                            │
│  Dropout (p=0.2)                                    │
│         ↓                                            │
│  Linear (1280→6)                                    │
│         ↓                                            │
│  Output (6 classes)                                 │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Key Specifications**:
- **Total Parameters**: 2,231,558
- **Trainable Parameters**: 2,231,558
- **Model Size (FP32)**: 8.75 MB
- **Architecture Highlights**: 
  - Depthwise separable convolutions
  - Inverted residual structure
  - Linear bottlenecks

### Edge Model Optimization

```
PyTorch Baseline (FP32)          ONNX Conversion          Dynamic Quantization
     89.20% acc          →    Cross-platform    →         INT8 weights
     8.75 MB                   optimization               88.50% acc
     45.32 ms                                             2.23 MB
                                                          12.85 ms
```

<details>
<summary><b>🔧 Training Configuration</b></summary>

```yaml
Optimizer:
  Type: Adam
  Learning Rate: 0.001
  Weight Decay: 0.0001

Scheduler:
  Type: ReduceLROnPlateau
  Patience: 3
  Factor: 0.5

Loss: CrossEntropyLoss
Batch Size: 32
Epochs: 25 (with early stopping)
Early Stopping Patience: 5

Data Augmentation:
  - Random Horizontal Flip (p=0.5)
  - Random Vertical Flip (p=0.3)
  - Random Rotation (±20°)
  - Color Jitter (brightness, contrast, saturation)
  - Random Affine
```

</details>

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Flutter SDK 3.0+ for mobile app
- (Optional) CUDA-capable GPU for training

### Step-by-Step Installation

<details open>
<summary><b>1️⃣ Clone the Repository</b></summary>

```bash
git clone https://github.com/yourusername/rice-disease-detection.git
cd rice-disease-detection
```

</details>

<details open>
<summary><b>2️⃣ Set Up Python Environment (Recommended)</b></summary>

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

</details>

<details open>
<summary><b>3️⃣ Install Dependencies</b></summary>

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `torch>=2.0.0` - Deep learning framework
- `onnxruntime>=1.15.0` - Optimized inference
- `numpy`, `pandas` - Data processing
- `pillow`, `opencv-python` - Image handling

</details>

<details>
<summary><b>4️⃣ Download Pre-trained Models (Optional)</b></summary>

Place the model files in the `models/` directory:
- `baseline_model.pth` (8.75 MB) - Baseline PyTorch model
- `edge_model.onnx` (2.23 MB) - Optimized ONNX model

Or train your own using the provided notebook!

</details>

<details>
<summary><b>5️⃣ Verify Installation</b></summary>

```python
# Quick test
python -c "import torch; import onnxruntime; print('✓ Installation successful!')"
```

</details>

---

## 💻 Usage

### 1️⃣ Single Image Prediction

```bash
python scripts/predict.py \
    --model models/edge_model.onnx \
    --image path/to/rice_leaf.jpg \
    --visualize
```

**Output**:
```
============================================================
PREDICTION RESULTS
============================================================
Predicted Disease: Brown Spot
Confidence: 94.32%
Inference Time: 12.85 ms
```

### 2️⃣ Batch Processing

```bash
python scripts/predict.py \
    --model models/edge_model.onnx \
    --input images_folder/ \
    --output results.csv \
    --return-probs
```

### 3️⃣ Model Evaluation

```bash
python scripts/evaluate.py \
    --model models/edge_model.onnx \
    --test-data data/test/ \
    --output evaluation_results/
```

**Generates**:
- ✅ Confusion matrix visualization
- ✅ Per-class performance metrics
- ✅ Detailed JSON report
- ✅ Text summary

### 4️⃣ Python API

```python
from scripts.predict import RiceLeafDiseaseDetector

# Initialize detector
detector = RiceLeafDiseaseDetector('models/edge_model.onnx')

# Single prediction
result = detector.predict('rice_leaf.jpg')
print(f"Disease: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = detector.predict_batch(images, return_probs=True)

# Get detailed probabilities
for img, res in zip(images, results):
    print(f"\n{img}:")
    print(f"  Prediction: {res['predicted_class']}")
    print(f"  Confidence: {res['confidence']:.2%}")
    print(f"  Top 3 predictions:")
    sorted_probs = sorted(
        res['all_probabilities'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:3]
    for disease, prob in sorted_probs:
        print(f"    - {disease}: {prob:.2%}")
```

### 5️⃣ Training from Scratch

```bash
# Open Jupyter notebook
cd notebooks
jupyter notebook rice_disease_training.ipynb

# Follow the step-by-step training pipeline
```

### 6️⃣ Model Conversion (PyTorch → ONNX)

```bash
python scripts/convert_to_onnx.py \
    --input baseline_model.pth \
    --output edge_model.onnx \
    --quantize \
    --test
```

---

## 📱 Mobile App

### Flutter Application

A **cross-platform mobile app** built with Flutter for real-time rice leaf disease detection.

<div align="center">

### 🎨 App Features

| Feature | Description |
|:--------|:------------|
| 📷 **Camera Integration** | Capture leaf images directly |
| 🖼️ **Gallery Support** | Select existing photos |
| 🤖 **Offline Inference** | No internet required |
| ⚡ **Real-time Results** | Sub-100ms predictions |
| 📊 **Detailed Analysis** | Probability distributions |
| 💾 **Lightweight** | Only 2.23 MB model |

</div>

### Quick Setup

```bash
# Navigate to mobile app directory
cd mobile_app

# Install Flutter dependencies
flutter pub get

# Run on connected device
flutter run --release
```

### Build for Release

```bash
# Android APK
flutter build apk --release

# Android App Bundle (Play Store)
flutter build appbundle --release

# iOS (requires Xcode)
flutter build ios --release
```

### Technical Stack

```
┌─────────────────────────────────────────┐
│        Mobile App Architecture           │
├─────────────────────────────────────────┤
│                                          │
│  📱 UI Layer (Flutter)                  │
│      └─ Material Design                 │
│                                          │
│  🎯 Business Logic                      │
│      └─ InferenceService (Provider)     │
│                                          │
│  🔧 ML Runtime                          │
│      └─ ONNX Runtime Mobile             │
│                                          │
│  📦 Plugins                             │
│      ├─ camera                          │
│      ├─ image_picker                    │
│      └─ image processing                │
│                                          │
└─────────────────────────────────────────┘
```

### Performance

- **Model Size**: 2.23 MB
- **Inference Time**: 50-100ms (mid-range devices)
- **Memory Usage**: <100 MB
- **Battery Impact**: Minimal
- **Platforms**: iOS 12.0+, Android API 21+

<details>
<summary><b>📸 Screenshots</b></summary>

| Screen | Description |
|:------:|:------------|
| ![Home](assets/mobile_home.png) | Home screen with capture options |
| ![Camera](assets/mobile_camera.png) | Camera interface |
| ![Result](assets/mobile_result.png) | Prediction results with confidence |
| ![Chart](assets/mobile_chart.png) | Probability distribution chart |

</details>

<details>
<summary><b>⚙️ Configuration</b></summary>

**Required Permissions**:

*Android (AndroidManifest.xml)*:
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

*iOS (Info.plist)*:
```xml
<key>NSCameraUsageDescription</key>
<string>Camera access needed to capture rice leaf images</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>Photo library access to select images</string>
```

</details>

---

## 📁 Project Structure

```
rice-disease-detection/
│
├── 📄 README.md                          # You are here
├── 📄 TECHNICAL_APPROACH.md              # Detailed methodology
├── 📄 PROJECT_SUMMARY.md                 # Executive summary
├── 📄 SETUP.md                           # Installation guide
├── 📄 QUICK_REFERENCE.md                 # Command cheat sheet
├── 📄 requirements.txt                   # Python dependencies
├── 📄 LICENSE                            # MIT License
├── 📄 .gitignore                         # Git ignore rules
│
├── 📓 notebooks/                         # Jupyter notebooks
│   └── rice_disease_training.ipynb      # Complete training pipeline
│
├── 🐍 scripts/                           # Python scripts
│   ├── predict.py                       # Inference script
│   ├── evaluate.py                      # Model evaluation
│   └── convert_to_onnx.py               # Model conversion
│
├── 🤖 models/                            # Saved models
│   ├── baseline_model.pth               # PyTorch baseline (8.75 MB)
│   └── edge_model.onnx                  # Optimized ONNX (2.23 MB)
│
├── 📱 mobile_app/                        # Flutter application
│   ├── lib/
│   │   ├── main.dart                    # App entry point
│   │   ├── screens/
│   │   │   ├── home_screen.dart
│   │   │   └── result_screen.dart
│   │   └── services/
│   │       └── inference_service.dart   # ONNX Runtime wrapper
│   ├── assets/
│   │   └── edge_model.onnx              # Model for mobile
│   ├── pubspec.yaml                     # Flutter dependencies
│   └── README.md                        # Mobile app docs
│
├── 📊 data/                              # Dataset (not in repo)
│   ├── train/
│   ├── val/
│   └── test/
│
└── 🖼️ assets/                            # Documentation assets
    ├── diagrams/
    ├── screenshots/
    └── results/
```

---

## 📊 Results

### Training Performance

```
┌──────────────────────────────────────────────────────┐
│           Baseline Model Training Results             │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Training Time:      ~45 minutes (Tesla T4 GPU)     │
│  Best Epoch:         6 / 25                           │
│  Final Train Acc:    86.33%                          │
│  Final Val Acc:      89.20% ⭐                       │
│  Test Accuracy:      89.20%                          │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### Per-Class Performance

<div align="center">

| Disease Class | Precision | Recall | F1-Score | Support |
|:--------------|:---------:|:------:|:--------:|:-------:|
| **Bacterial Leaf Blight** | 0.91 | 0.89 | 0.90 | 95 |
| **Brown Spot** | 0.88 | 0.87 | 0.87 | 97 |
| **Healthy Rice Leaf** | 0.94 | 0.95 | 0.94 | 98 |
| **Leaf Blast** | 0.86 | 0.88 | 0.87 | 95 |
| **Leaf Scald** | 0.87 | 0.86 | 0.86 | 94 |
| **Sheath Blight** | 0.89 | 0.90 | 0.89 | 96 |
| | | | | |
| **Overall Accuracy** | | | **89.20%** | 575 |
| **Macro Average** | 0.89 | 0.89 | 0.89 | 575 |
| **Weighted Average** | 0.89 | 0.89 | 0.89 | 575 |

</div>

### Inference Speed Comparison

```
Hardware: Intel Core i7-9750H @ 2.60GHz (CPU only)

                Baseline (PyTorch)     Edge (ONNX)      Improvement
              ┌────────────────────┬─────────────────┬──────────────┐
Mean Time     │   45.32 ms         │   12.85 ms      │   ↓ 71.6%   │
Std Dev       │    2.15 ms         │    0.87 ms      │   ↓ 59.5%   │
Throughput    │   ~22 FPS          │   ~78 FPS       │   ↑ 3.5x    │
              └────────────────────┴─────────────────┴──────────────┘
```

### Confusion Matrix

```
Predicted →
True ↓         Bact  Brown  Healthy  Blast  Scald  Sheath
Bacterial       85     3       1       4      1      1
Brown Spot       2    84       0       5      4      2
Healthy          0     1      93       2      1      1
Leaf Blast       3     4       1      84      2      1
Leaf Scald       2     3       2       3     81      3
Sheath Blight    1     2       1       2      4     86
```

### Model Size Breakdown

```
┌────────────────────────────────────────────┐
│         Component Size Analysis             │
├────────────────────────────────────────────┤
│                                             │
│  Baseline (PyTorch FP32):                  │
│  ├─ Model Weights:     8.50 MB             │
│  ├─ Metadata:          0.25 MB             │
│  └─ Total:             8.75 MB             │
│                                             │
│  Edge (ONNX INT8):                         │
│  ├─ Quantized Weights: 2.13 MB (-75%)     │
│  ├─ Graph Structure:   0.10 MB             │
│  └─ Total:             2.23 MB (-74.5%)   │
│                                             │
└────────────────────────────────────────────┘
```

### ROI Analysis

For **only 0.7%** accuracy sacrifice:

```
✅ Benefits Gained:
   • 6.52 MB size reduction (mobile-friendly)
   • 32.47 ms faster inference (better UX)
   • 3.5x throughput improvement
   • Lower power consumption (battery savings)
   • Offline capability (privacy + convenience)
   • Zero server costs (complete edge deployment)

❌ Cost Paid:
   • 0.7% accuracy drop (88.5% still excellent)

📈 Value Proposition: 106:1 benefit-to-cost ratio
```

---

## 📚 Documentation

Comprehensive documentation is available:

| Document | Description | Link |
|:---------|:------------|:----:|
| 📖 **README** | Quick start and overview | [README.md](README.md) |
| 🔬 **Technical Approach** | Detailed methodology (40+ pages) | [TECHNICAL_APPROACH.md](TECHNICAL_APPROACH.md) |
| 📋 **Project Summary** | Executive overview | [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) |
| ⚡ **Quick Reference** | Command cheat sheet | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |
| 🛠️ **Setup Guide** | Installation instructions | [SETUP.md](SETUP.md) |
| 📱 **Mobile App Docs** | Flutter app guide | [mobile_app/README.md](mobile_app/README.md) |

### 📖 What Each Document Covers

<details>
<summary><b>TECHNICAL_APPROACH.md</b> - Deep Technical Dive</summary>

- Problem statement and requirements
- Dataset analysis and preprocessing
- Model architecture selection rationale
- Training strategy and hyperparameters
- Optimization techniques (ONNX, quantization)
- Mobile deployment architecture
- Evaluation methodology
- Challenges and solutions
- Results and trade-off analysis

</details>

<details>
<summary><b>PROJECT_SUMMARY.md</b> - Executive Overview</summary>

- Key achievements and metrics
- Technical architecture overview
- Skills demonstrated
- Deliverables checklist
- Performance benchmarks
- Future improvements

</details>

<details>
<summary><b>QUICK_REFERENCE.md</b> - Command Cheat Sheet</summary>

- Essential commands
- API usage examples
- Troubleshooting tips
- Performance optimization
- Hardware specifications

</details>

---

## 🔮 Future Improvements

### 🎯 Short-term Enhancements

- [ ] **Model Improvements**
  - Test EfficientNet and Vision Transformers
  - Implement knowledge distillation
  - Apply pruning techniques for additional compression
  
- [ ] **Dataset Expansion**
  - Collect images with diverse lighting conditions
  - Add more environmental variations
  - Implement synthetic data generation

### 🚀 Medium-term Goals

- [ ] **Deployment Options**
  - TensorFlow Lite conversion
  - Web interface (WebAssembly/ONNX.js)
  - REST API for cloud inference
  - Edge TPU optimization for Coral devices

- [ ] **Feature Additions**
  - Multi-language support (10+ languages)
  - Disease progression tracking
  - Treatment recommendations database
  - Offline help documentation

### 🌟 Long-term Vision

- [ ] **Advanced Features**
  - Multi-disease detection per image
  - Severity assessment
  - Weather integration for risk prediction
  - Community platform for farmers

- [ ] **Optimization**
  - INT4 quantization experiments
  - Neural architecture search
  - Hardware-specific optimizations (NEON, AVX2)

---

## 🤝 Contributing

Contributions are **welcome**! We appreciate all contributions, from bug fixes to new features.

### How to Contribute

1. **Fork the Repository**
   ```bash
   # Click 'Fork' button on GitHub
   ```

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/yourusername/rice-disease-detection.git
   cd rice-disease-detection
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

4. **Make Your Changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests if applicable

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: Amazing new feature"
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/AmazingFeature
   ```

7. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Describe your changes

### Contribution Guidelines

- ✅ Write clear commit messages
- ✅ Update documentation as needed
- ✅ Test your changes thoroughly
- ✅ Follow Python PEP 8 style guide
- ✅ Add comments for complex logic
- ✅ Update requirements.txt if needed

### Areas We Need Help

🐛 **Bug Reports**: Found a bug? Open an issue!  
📝 **Documentation**: Help improve docs  
🌍 **Translations**: Multi-language support  
🎨 **UI/UX**: Mobile app improvements  
🧪 **Testing**: Add more test cases  
🚀 **Features**: Implement new capabilities  

### Code of Conduct

Be respectful and inclusive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND...
```

### What This Means

✅ **You CAN**:
- Use for commercial purposes
- Modify the code
- Distribute copies
- Use privately
- Sublicense

❌ **You MUST**:
- Include license and copyright notice
- State changes made to the code

⚠️ **You CANNOT**:
- Hold authors liable
- Expect warranty

---

## 👤 Author

**[Your Name]**

- 🌐 GitHub: [@yourusername](https://github.com/yourusername)
- 💼 LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)
- 📧 Email: your.email@example.com
- 🐦 Twitter: [@yourhandle](https://twitter.com/yourhandle)

---

## 🙏 Acknowledgments

Special thanks to:

- **Kaggle** - For providing the rice disease dataset
- **PyTorch Team** - For the excellent deep learning framework
- **ONNX Community** - For standardized model format
- **Flutter Team** - For cross-platform mobile framework
- **ONNX Runtime Team** - For optimized inference engine
- **MobileNetV2 Authors** - For mobile-optimized architecture
- **Open Source Community** - For making this project possible

### Research References

1. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks" - [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)
2. ONNX: Open Neural Network Exchange - [onnx.ai](https://onnx.ai/)
3. PyTorch Quantization Documentation - [pytorch.org/docs](https://pytorch.org/docs/stable/quantization.html)

---

## 📊 Project Stats

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/yourusername/rice-disease-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/rice-disease-detection?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/rice-disease-detection)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/rice-disease-detection)
![GitHub license](https://img.shields.io/github/license/yourusername/rice-disease-detection)

**Made with ❤️ for the agricultural community**

</div>

---

## 📞 Support

Need help? Have questions?

- 📖 Check the [Documentation](#-documentation)
- 🐛 [Open an Issue](https://github.com/yourusername/rice-disease-detection/issues)
- 💬 [Start a Discussion](https://github.com/yourusername/rice-disease-detection/discussions)
- 📧 Email: your.email@example.com

---

## ⭐ Show Your Support

If you found this project helpful, please consider:

- ⭐ **Starring** this repository
- 🐛 **Reporting** bugs and issues
- 💡 **Suggesting** new features
- 🔀 **Contributing** code improvements
- 📢 **Sharing** with others who might benefit

---

<div align="center">

### 🌾 Help farmers detect rice diseases early and improve crop yields! 🌾

**Built for the Edge AI Engineer position at Klyff**

*Last Updated: February 2026*

</div>
