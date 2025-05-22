# -Signature-Forgery-Detection-using-Siamese-Network-CEDAR-Dataset-
This project implements a Signature Forgery Detection System using a Siamese Neural Network trained on the CEDAR dataset. It uses PyTorch Lightning for training and Gradio for a simple web interface to test signature authenticity

## 🚀 Features

* Siamese Network using pretrained `ResNet18` backbone
* Contrastive loss for learning signature similarity
* Trained on CEDAR signature dataset (`original` and `forged` images)
* Gradio UI to test signatures interactively
* Easily extendable and customizable

---

## 🗂️ Dataset Structure (CEDAR Format)

Expected directory structure:

```
dataset/
├── original/
│   ├── 001/
│   │   ├── original_001_01.png
│   │   └── ...
├── forged/
│   ├── 001/
│   │   ├── forged_001_01.png
│   │   └── ...
```

> You can download the CEDAR dataset from [CEDAR Signature Dataset](http://www.cedar.buffalo.edu/NIJ/data/signatures.rar)

---

## 🧠 Model Overview

This Siamese Network uses ResNet18 (without final classification head) as the feature encoder. It compares two signatures using the L2 distance between encoded vectors and learns to minimize distance for genuine pairs and maximize it for forgeries.

Loss Function:

```math
L = y * D^2 + (1 - y) * max(0, margin - D)^2
```

Where:

* `D` is the L2 distance between signature embeddings
* `y = 1` for genuine pairs, `0` for forged
* `margin` is a hyperparameter (default: 1.0)

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/signature-forgery-detection.git
cd signature-forgery-detection

# Install dependencies
pip install torch torchvision torchmetrics pytorch-lightning
pip install transformers datasets
pip install gradio
```

---

## 🧪 Training

Update the `path` variable in the code with your dataset path:

```python
path = "/path/to/your/cedar/dataset"
```

Then run the script to:

* Prepare dataset
* Train the Siamese Network
* Save the model to `siamese_signature_model.pth`

```bash
python siamese_signature_detector.py
```

> ⚠️ Ensure the dataset folders `original` and `forged` exist inside the specified path.

---

## 🖥️ Gradio Interface

Once trained, the script will automatically load the model and launch a Gradio interface.

You can:

* Upload an **authentic signature**
* Upload a **test signature**
* Get a prediction (Authentic or Forged)

```bash
# Interface launch (automatically triggered at the end of the script)
Launching Gradio interface...
```

Or manually launch the interface:

```python
iface.launch()
```

---

## 📊 Inference Example

Example output after comparing two signatures:

```
Signatures are similar (likely Authentic). Distance: 0.2641
```

Or:

```
Signatures are dissimilar (likely Forged). Distance: 1.0312
```

> You may adjust the threshold (`0.5`) based on validation/test performance.

---

## 📁 File Structure

```
.
├── siamese_signature_detector.py  # Main training + interface script
├── siamese_signature_model.pth    # Saved model (after training)
└── README.md
```

---

## ✅ TODO

* [ ] Add validation & testing pipeline
* [ ] Automatically compute optimal threshold from ROC-AUC
* [ ] Deploy on Hugging Face Spaces
* [ ] Add support for other signature datasets (e.g. GPDS, BHSig260)

---

## 🤝 Credits

* **CEDAR Dataset** – [Center of Excellence for Document Analysis and Recognition](http://www.cedar.buffalo.edu/)
* Inspired by various Siamese Network implementations in PyTorch
* Built with ❤️ using [PyTorch Lightning](https://www.pytorchlightning.ai/) and [Gradio](https://www.gradio.app/)

---

## 📜 License

This project is open-source and free to use under the [MIT License](LICENSE).

