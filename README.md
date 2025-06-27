# 🐾 Pet Recognition with CNNs & Pre-trained Models

This project is a practical exploration of **Convolutional Neural Networks (CNNs)** and **transfer learning using pre-trained models**.
It's part of my journey into learning artificial intelligence and deep learning through hands-on experimentation.

The goal is to classify different pet breeds using image data — specifically, the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — while
understanding model performance, overfitting, fine-tuning, and model evaluation.

---

## Project Structure

- `main.py` — Main training and evaluation script
- `src/` — Source code (models, data loading, training)
- `models/` — Saved model weights (`.pth` files)
- `notebooks/analysis.ipynb` — Analysis and comparison of selected models
- `experiments.md` — Log of experiments and results
<!-- - `ui.py` — Gradio-based web interface to interactively test the trained models-->

---

## Dataset

- **Oxford-IIIT Pet Dataset**  
  Contains 37 categories with roughly 200 images per class, labeled with breed and species (cat or dog).

---

## Results & Analysis

- Experimental logs and model performances: [`experiments.md`](./experiments.md)
- Model comparison & confusion matrices: [`notebooks/analysis.ipynb`](./notebooks/analysis.ipynb)

---

<!--## 🌐 Live Demo

A simple Gradio interface is available to test the top 7 trained models interactively. To launch it:

```bash
python main.py --ui
```

You can select a trained model, upload an image, and view predictions with confidence scores.

---

## 💡 Future Improvements

- Add model ensembling
- Improve dataset augmentations
- Hyperparameter optimization
- Support ONNX export or deployment

---
-->
Feel free to fork, explore, or contribute!  
This project is meant for learning and experimentation 🌱
