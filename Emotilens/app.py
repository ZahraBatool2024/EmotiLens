from pathlib import Path
import re
import joblib
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

MODEL_DIR = Path(
    r"C:\Users\PMY\Desktop\New folder (2)\sentiment_model_output"
)

MODEL_PATH = MODEL_DIR / "linear_svm_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
LABELS_PATH = MODEL_DIR / "label_mapping.joblib"

for resource in ('stopwords', 'wordnet', 'omw-1.4'):
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

try:
    STOP_WORDS = set(stopwords.words('english')) - {'no', 'not', 'nor'}
except LookupError as exc:
    raise RuntimeError("NLTK resources missing. Run: python -c \"import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')\"") from exc
lemmatizer = WordNetLemmatizer()

missing = [p.name for p in (MODEL_PATH, VECTORIZER_PATH, LABELS_PATH) if not p.exists()]
if missing:
    raise FileNotFoundError(f"Missing model files {missing}. Copy your trained sentiment_model_output folder next to app.py.")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_names = joblib.load(LABELS_PATH)
app = Flask(__name__)

META = {
    'sadness': ('🌧️', 'This text seems to express sadness or disappointment.'),
    'joy': ('☀️', 'This text seems to express happiness or excitement.'),
    'love': ('💗', 'This text seems to express affection or fondness.'),
    'anger': ('🔥', 'This text seems to express frustration or anger.'),
    'fear': ('🌙', 'This text seems to express worry or fear.'),
    'surprise': ('✨', 'This text seems to express surprise or amazement.'),
}

def preprocess_text(text):
    tokens = re.findall(r'[a-z]+', str(text).lower())
    return ' '.join(lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS)

def classify(text):
    cleaned = preprocess_text(text)
    if not cleaned.strip():
        return None
    features = vectorizer.transform([cleaned])
    label = int(model.predict(features)[0])
    sentiment = str(label_names[label]).lower()
    emoji, description = META.get(sentiment, ('💬', 'The model has made a prediction.'))
    raw = model.decision_function(features)
    if getattr(raw, 'ndim', 1) == 1:
        values = raw
    else:
        values = raw[0]
    scores = [{'name': str(label_names[int(cls)]).capitalize(), 'key': str(label_names[int(cls)]).lower(), 'score': round(float(values[i]), 4)} for i, cls in enumerate(model.classes_)]
    scores.sort(key=lambda item: item['score'], reverse=True)
    return {'sentiment': sentiment.capitalize(), 'sentiment_key': sentiment, 'emoji': emoji,
            'description': description, 'cleaned_text': cleaned, 'scores': scores}

@app.get('/')
def home():
    return render_template('index.html')

@app.post('/predict')
def predict():
    data = request.get_json(silent=True) or request.form
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify(error='Please enter a sentence to analyze.'), 400
    if len(text) > 2000:
        return jsonify(error='Please keep your text under 2,000 characters.'), 400
    result = classify(text)
    if result is None:
        return jsonify(error='Please enter meaningful English words.'), 400
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
