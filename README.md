# GameFace-AI: Multiverse Character Classifier

![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)

**GameFace-AI** is an evolving computer vision project designed to identify and categorize iconic video game characters. By leveraging deep learning and advanced image augmentation, this project aims to create a robust model capable of distinguishing between high-similarity characters across different franchises (e.g., *The Witcher*, *Sekiro: Shadows Die Twice*).

## 🗺️ The Roadmap: From Witcher to Shinobi

This project is built in iterative stages, following a "crawl, walk, run" philosophy to tackle increasing levels of complexity.

### ✅ Stage 1: The Witcher Binary (Complete)

**Goal:** Distinguish Geralt of Rivia from all other game elements (NPCs, environments, monsters).

* **Architecture:** ResNet-18 (backbone unfrozen for fine-tuning).
* **Key Challenge:** The "Vesemir Bias." The model initially struggled with Vesemir due to shared Witcher features (white hair, orange eyes).
* **Breakthrough:** Overcame "Perspective Bias" (the model failing on cinematic camera angles) by implementing `RandomPerspective` transforms.
* **Result:** **91% Validation Accuracy** on unseen data.

### 🏗️ Stage 2: Multi-Class Identity (Current Focus)

**Goal:** Specific identification of multiple characters within the same engine.

* **Target Classes:** `Geralt`, `Ciri`, `Yennefer`, `Vesemir`, `Triss`, `Other`
* **Objective:** Move beyond "Not-Geralt" to specific labeling, training the model to recognize distinct facial landmarks for each hero.
* **The V6 Breakthrough:** Early iterations suffered from 57% Precision for Vesemir, as the model confused him with bearded/scarred NPCs (like Damien de la Tour or King Bran). Implementing Hard Negative Mining by manually curating the Other training set to include "imposter" characters (Damien, King Bran, Regis) to force the model to learn fine-grained facial landmarks over generic traits solved the problem.
* **Some performance metrics:**
     * **Overall accuracy:** 85,31%
     * **Yennefer:** Achieving near-perfect performance (95% accuracy, 100% precision, 95% recall and 97% F1-score).
     * **Robustness:** Successfully identified new, unseen NPCs in the validation set.

### 🚀 Stage 3: The Multiverse Expansion (Upcoming)

**Goal:** Cross-game character recognition.

* **Planned Additions:**
    * **Sekiro:** Wolf (Sekiro), Isshin Ashina, Genichiro.
    * **Future:** Iconic characters from various titles.
* **Tech Shift:** Implementing **Object Detection (YOLO)** to handle multiple characters from different art styles in the same frame.

---

## 🛠️ Setup & Dataset Structure

### 1. Requirements

```bash
pip install torch torchvision pillow matplotlib
```

### 2. Organizing your Data

To run the training scripts, organize your local `data/` folder as follows:

```
data/
├── train/
│   ├── geralt/
│   ├── vesemir/
│   ├── .....
│   ├── wolf_sekiro/  # Placeholder for Stage 3
│   └── background/   # Environmental shots, guards, and UI
└── val/              # Mirrored structure for testing
```

## 🧠 Key Technical Insights

* **Unfrozen Backbones:** Fine-tuning the ResNet-18 backbone was crucial for learning game-specific textures.
* **Robustness Testing:** The model has been verified against horizontal flips and center-crops to ensure it recognizes facial features rather than background context.
* **Perspective Warping:** Simulating low-angle views during training fixed the model's primary blind spot in cinematic screenshots.

## ⚖️ Legal

This is an unofficial fan project. All character designs and game assets are the property of their respective creators (CD PROJEKT RED, FromSoftware, etc.).
