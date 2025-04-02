# 🧠 GoogLeNet Transfer Learning: CINIC-10 ➜ CIFAR-10

This project implements a modular image classification pipeline using **GoogLeNet with Batch Normalization** in PyTorch. The model is first **pretrained on the CINIC-10 dataset**, then **fine-tuned on CIFAR-10**, with a **from-scratch training baseline** for comparison.

---

## 📌 Features

- ✅ Modular training pipeline with CLI via `main.py`
- ✅ Pretraining on CINIC-10
- ✅ Fine-tuning on CIFAR-10 using pretrained weights
- ✅ Training from scratch on CIFAR-10
- ✅ Best model saving and early stopping
- ✅ TensorBoard logging support

---

## 📁 Project Structure

```
GoogLeNet_CINIC10_CIFAR10/
│
├── main.py               # Unified entry point (train / finetune / scratch)
├── train.py              # Pretraining on CINIC-10
├── finetune.py           # Fine-tuning on CIFAR-10 using pretrained weights
├── scratch.py            # Training from scratch on CIFAR-10
│
├── models/
│   └── googlenet_bn.py   # GoogLeNet with BatchNorm
│
├── data_utils.py         # Data loading and augmentation
├── configs.py            # Central configuration file
├── utils.py              # Training loop, evaluation, logging, saving
│
├── checkpoints/          # Saved models (auto-created)
├── runs/                 # TensorBoard logs
├── CINIC-10/             # CINIC-10 dataset (downloaded and extracted)
├── data/                 # CIFAR-10 download cache
└── requirements.txt      # Python dependencies
```

---

## ⚙️ Setup

```bash
conda create -n googlenet_cinic10 python=3.9
conda activate googlenet_cinic10
pip install -r requirements.txt
```

---

## 📥 Dataset Preparation

### CINIC-10
1. Download from: [CINIC-10 Official Download](https://datashare.ed.ac.uk/handle/10283/3192)  
2. Extract into the project root directory as `./CINIC-10/`  
   (Should contain `train/`, `valid/`, `test/` subfolders)

### CIFAR-10
No need to manually download. It will be automatically downloaded via `torchvision`.

---

## 🚀 How to Use

Run via the unified entry point `main.py` with one of three modes:

### 1. 🧠 Pretraining on CINIC-10
```bash
python main.py --mode train
```

- Uses: `train.py`
- Saves best model to: `checkpoints/googlenet_cinic10_best.pth`

---

### 2. 🔧 Fine-tuning on CIFAR-10
```bash
python main.py --mode finetune
```

- Uses: `finetune.py`
- Loads pretrained weights from CINIC-10
- Trains only the final `fc` layer
- Saves best model to: `checkpoints/googlenet_finetuned_best.pth`

---

### 3. 🧪 From-Scratch Training on CIFAR-10
```bash
python main.py --mode scratch
```

- Uses: `scratch.py`
- Trains all layers from random initialization
- Saves best model to: `checkpoints/googlenet_scratch_best.pth`

---

## 📈 TensorBoard Visualization

```bash
tensorboard --logdir runs
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

---

## 📊 Sample Results (for reference)

| Method          | Pretrained | Test Accuracy | Notes                  |
|-----------------|------------|---------------|------------------------|
| Pretrained      | CINIC-10   | ~74%          | Used as fine-tune base |
| Fine-tuned      | ✅ Yes      | **82%+**      | Fast convergence       |
| From Scratch    | ❌ No       | ~78% (±1%)    | Slower to converge     |

---

## 🔮 Future Work

- [ ] Add `test.py` for evaluation on CINIC-10 test set
- [ ] Add model inference on custom images
- [ ] Support for learning rate scheduling (e.g., cosine decay)
- [ ] Extend to CIFAR-100 or Tiny-ImageNet

---
