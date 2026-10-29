<div align=center>
    
# EmotiLens

**Emotion classification from text, powered by TF-IDF and a Linear SVM.**

</div>

EmotiLens is a machine-learning web app that predicts the emotion expressed in a piece of text. It classifies input into six categories: **sadness, joy, love, anger, fear, and surprise**. A Flask backend serves the trained model through a clean, dark-themed web interface.

<div align=center>
    
## Features
</div>

- Predicts one of six emotions from user-provided text
- TF-IDF features (unigrams and bigrams) with a LinearSVC classifier
- Text preprocessing: stop-word removal and WordNet lemmatization
- Class-balanced training data (downsampled to the smallest class)
- Evaluation with accuracy, precision, recall, F1-score, classification report, and confusion matrix
- Interactive UI showing the predicted emotion and a score comparison across all six classes

> **Note:** Displayed scores are SVM decision margins, not calibrated probabilities. They do not represent the percentage likelihood that an emotion is correct.

<div align=center>
    
## Emotion Labels

| Label | Emotion  |
|:-----:|----------|
| 0     | Sadness  |
| 1     | Joy      |
| 2     | Love     |
| 3     | Anger    |
| 4     | Fear     |
| 5     | Surprise |

## Tech Stack

| Area            | Tools                                  |
|-----------------|----------------------------------------|
| Language        | Python                                 |
| Machine learning| scikit-learn (`TfidfVectorizer`, `LinearSVC`) |
| Preprocessing   | NLTK (stop words, WordNet lemmatizer)  |
| Backend         | Flask                                  |
| Frontend        | HTML, CSS, JavaScript                  |
| Persistence     | Joblib                                 |

## Project Structure
</div>

```text
EmotiLens/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── app.js
└── sentiment_model_output/
    ├── linear_svm_model.joblib
    ├── tfidf_vectorizer.joblib
    └── label_mapping.joblib
```
<div align=center>
    
## Dataset and Training Pipeline
</div>

The dataset contains `text` and `label` columns, with labels 0–5 as listed above.

1. Load the data and remove missing and duplicate records
2. Clean text, remove stop words, and lemmatize
3. Balance classes by downsampling to the smallest class
4. Split into train and test sets (stratified 70/30)
5. Extract TF-IDF features (unigrams and bigrams)
6. Train a Linear SVM classifier
7. Evaluate with classification metrics and a confusion matrix

<div align=center>
    
## Installation
</div>

**1. Clone the repository**

```bash
git clone https://github.com/ZahraBatool2024/EmotiLens.git
cd EmotiLens
```

**2. Create and activate a virtual environment**

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The model was trained with **scikit-learn 1.4.2**; using the same version is recommended for compatibility:

```bash
python -m pip install scikit-learn==1.4.2 flask joblib nltk
```

**4. Download NLTK resources (one time)**

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Usage

Ensure the model files are in the directory configured in `app.py` (update the path there if yours differs):

- `linear_svm_model.joblib`: trained classifier
- `tfidf_vectorizer.joblib`: fitted TF-IDF vectorizer
- `label_mapping.joblib`: label-to-emotion mapping

Start the app:

```bash
python app.py
```

Then open <http://127.0.0.1:5000>, enter a sentence, and submit it to see the predicted emotion and score chart.

<div align= center>
    
## Limitations and Notes
</div>

- The app loads the saved model and vectorizer; it does not train at startup.
- Keep preprocessing identical between training and prediction. If you change stop-word handling or lemmatization, retrain both the vectorizer and classifier and re-save the artifacts.
- TF-IDF with a Linear SVM relies on learned word patterns and may misclassify sarcasm, negation, context-dependent meaning, or unfamiliar phrasing.
- Flask's development server is for local use only. For deployment, use a production WSGI server and appropriate security settings.
- Avoid committing private data or large model files to a public repository unless you have permission and your host's size limits allow it.

