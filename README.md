# Plant Leaf Disease Classification Web App

A Streamlit web application for a trained CNN / MobileNetV2 model.

## 1. Project files

- `app.py` — web interface for image upload and prediction
- `train.py` — trains a MobileNetV2 transfer-learning model
- `requirements.txt` — Python dependencies
- `models/` — trained model and class-name file are generated here
- `data/` — create train, val, and test folders here

## 2. Dataset folder layout

Use the same class folders in all three splits. For the starter project, the guideline suggests three tomato classes:

```text
data/
  train/
    Tomato_healthy/
    Tomato_Early_blight/
    Tomato_Late_blight/
  val/
    Tomato_healthy/
    Tomato_Early_blight/
    Tomato_Late_blight/
  test/
    Tomato_healthy/
    Tomato_Early_blight/
    Tomato_Late_blight/
```

Put your actual labeled images in the corresponding folders. Suggested split: 70% train, 15% validation, 15% test. Keep test images separate until final evaluation.

## 3. Set up on Windows

Open Command Prompt in this project folder. Use a Python version supported by the TensorFlow release you install. Then run:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow fails to install, check the Python/TensorFlow compatibility for your machine or use Google Colab to train the model.

## 4. Train the model

After preparing the data folders:

```bash
python train.py
```

The training script creates:

- `models/plant_disease.keras`
- `models/class_names.json`
- `models/accuracy.png`
- `models/confusion_matrix.png`

The first run may download pretrained MobileNetV2 weights and needs internet access. Training time and accuracy depend on the dataset and hardware.

## 5. Launch the web application

From the project root:

```bash
streamlit run app.py
```

Streamlit will show a local URL, usually `http://localhost:8501`. Open it in your browser, upload a JPG/JPEG/PNG leaf image, and click **Predict Disease**.

## Important

The app only predicts among the classes used to train the model. A high model score does not guarantee a correct real-world diagnosis, especially for field images, unfamiliar plants, or diseases not represented in the training set. Do not report invented accuracy; use the test results from your run.
