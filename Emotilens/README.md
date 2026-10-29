# EmotiLens — Six-Class Sentiment Website

Responsive Flask website using your saved TF-IDF + Linear SVM model. It does not retrain the model. The interface shows raw LinearSVC decision scores, **not probabilities**.

## 1. Copy your saved model files
Copy the `sentiment_model_output` folder created by your notebook into this project. It must contain:

- `linear_svm_model.joblib`
- `tfidf_vectorizer.joblib`
- `label_mapping.joblib`

Folder structure:

```text
sentiment_ai_website/
├── app.py
├── requirements.txt
├── sentiment_model_output/
│   ├── linear_svm_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── label_mapping.joblib
├── templates/index.html
└── static/{style.css,app.js}
```

## 2. Run in VS Code (Windows PowerShell)
Open this folder in VS Code, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

If virtual-environment activation is blocked, use ` .\.venv\Scripts\python.exe -m pip install -r requirements.txt` and ` .\.venv\Scripts\python.exe app.py`.

Open **http://127.0.0.1:5000** in your browser. Stop the server with `Ctrl+C`.

The first run may download NLTK resources and needs internet access. If needed, run:

```powershell
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Important: use the three artifacts from the same training run and keep the website preprocessing identical to the notebook. Flask's debug server is for local development only, not public production hosting.
