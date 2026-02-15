<div align="center">

# 🌾 Rice Leaf Disease Detection
### Edge AI Solution for Agricultural Disease Classification
### 🎯 An end-to-end Edge AI solution for rice leaf disease detection, optimized for mobile deployment with **74.5% size reduction** and **only 0.7% accuracy loss**.

</div>


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
+ 95.46% baseline accuracy on test set
+ 95.46% edge model accuracy (0% drop)
+ Full offline capability for mobile deployment
```

---

**That's it!** 🎉 You should see the disease prediction with confidence score.

---

## 📊 Trade-off Analysis

<div align="center">

### **Baseline vs Edge Model Comparison**

| Metric | Baseline Model | Edge Model | Change |
|:-------|:--------------:|:----------:|:------:|
| **📊 Accuracy** | 95.46% | 95.46% | <span style="color: orange;"> 0%</span> |
| **📱 Mobile Ready** | ❌ | ✅ | Yes |

</div>

### 💡 Why This Trade-off is Excellent

- ✅ **3.5x faster inference** - Better user experience
- ✅ **74.5% smaller model** - Fits easily in mobile apps
- ✅ **Lower power consumption** - Extended battery life
- ✅ **Offline capability** - Works anywhere, anytime
- ✅ **Privacy preserved** - Data never leaves device
- ✅ **No server costs** - Complete edge deployment

<details>

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
