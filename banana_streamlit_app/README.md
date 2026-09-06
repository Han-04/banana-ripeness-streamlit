# Banana Ripeness Streamlit App

This repository deploys the trained GrabCut + adaptive CIELAB K-means + SVM
banana-ripeness classifier as a Streamlit application.

## Repository structure

```text
banana_streamlit_app/
├── .streamlit/
│   └── config.toml
├── models/
│   ├── README.md
│   └── grabcut_cielab_kmeans_svm.joblib  # Add this trained file
├── preprocessing.py
├── requirements.txt
├── streamlit_app.py
└── README.md
```

## 1. Download the trained model from Colab

After running the training notebook, run:

```python
from google.colab import files

files.download(
    "/content/banana_svm_outputs/grabcut_cielab_kmeans_svm.joblib"
)
```

Place the downloaded file inside the `models` folder. Do not rename it unless
you also update `MODEL_PATH` in `streamlit_app.py`.

## 2. Match the scikit-learn version

A joblib SVM should be loaded with the same scikit-learn version used to train
it. Check the Colab version:

```python
import sklearn
print(sklearn.__version__)
```

If Colab prints, for example, `1.7.2`, change the corresponding line in
`requirements.txt` to:

```text
scikit-learn==1.7.2
```

## 3. Test locally

From the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 4. Upload to GitHub

Create a GitHub repository and upload the complete contents of this directory,
including the `.streamlit` directory and trained `.joblib` file.

## 5. Deploy on Streamlit Community Cloud

1. Open <https://share.streamlit.io> and connect the GitHub account.
2. Select **Create app**.
3. Select the repository and branch.
4. Set the entrypoint to `streamlit_app.py`.
5. In advanced settings, choose the Python version closest to the Colab runtime.
6. Select **Deploy**.

## Important model constraint

`preprocessing.py` must remain synchronized with the Colab training notebook.
Changing the image size, GrabCut settings, K candidates, histogram bins, or
feature order requires retraining and re-exporting the SVM.

## Confidence scores

The Streamlit app displays the probability assigned to the predicted class and
warns when it is below the saved 60% threshold. Run the updated Colab notebook
to train the SVM with `probability=True`, then replace the joblib file in
`models/`. An older model will still predict, but the app will show decision
scores instead of a confidence percentage.
