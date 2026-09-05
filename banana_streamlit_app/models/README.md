# Model location

Place the trained Colab model at exactly:

```text
models/grabcut_cielab_kmeans_svm.joblib
```

Download it from Colab with:

```python
from google.colab import files

files.download(
    "/content/banana_svm_outputs/grabcut_cielab_kmeans_svm.joblib"
)
```

Commit the `.joblib` file to the GitHub repository before deploying.
