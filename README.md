# Banana Ripeness Streamlit Application

## Files

- `streamlit_app.py` — GUI and the complete inference pipeline.
- `requirements.txt` — Python dependencies.
- `banana_ripeness_color_svm.joblib` — trained model bundle produced by the Colab notebook. You must add this file after training.

## Prepare the model

1. Open `banana_ripeness_streamlit_confidence_colab.ipynb` in Google Colab.
2. Run all cells through model training and evaluation.
3. Download `/content/banana_svm_artifacts/banana_ripeness_color_svm.joblib`.
4. Put the downloaded model in the same directory as `streamlit_app.py`.

The GUI also allows the model bundle to be uploaded through its sidebar. Only load a joblib file that you created or trust.

## Run locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`.

## Prediction output

For each uploaded image, the application displays:

- the predicted ripeness class;
- model confidence for the predicted class;
- probabilities for all four classes;
- the original image;
- the LAB color-segmentation mask;
- the segmented banana region of interest.

Confidence comes from `SVC.predict_proba`. It represents the classifier's relative certainty and is not a guarantee that the prediction is correct. Low-confidence results should be reviewed, especially when the image contains several objects, unusual lighting or a cluttered background.
