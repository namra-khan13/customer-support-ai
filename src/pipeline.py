import re
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ticket_classifier.joblib"

TRAINING_DATA = [
    ("I was charged twice for my order", "billing"),
    ("Please explain this extra payment on my invoice", "billing"),
    ("I need a refund for the amount deducted", "billing"),
    ("My payment failed but money was deducted", "billing"),
    ("The invoice amount is incorrect", "billing"),
    ("I cannot log in to my account", "technical"),
    ("The application crashes whenever I open it", "technical"),
    ("I get an error while resetting my password", "technical"),
    ("The website is not loading", "technical"),
    ("My account page is showing a blank screen", "technical"),
    ("Where is my package? It has not arrived", "shipping"),
    ("My order is delayed and tracking is not updating", "shipping"),
    ("The delivery person did not come today", "shipping"),
    ("I received the wrong delivery address update", "shipping"),
    ("Please change my shipping address", "shipping"),
    ("I want to cancel my order", "returns"),
    ("How can I return this damaged product?", "returns"),
    ("I received a defective item and want an exchange", "returns"),
    ("Please tell me your return policy", "returns"),
    ("I need to replace the product I purchased", "returns"),
    ("How do I update my phone number?", "account"),
    ("Please close my account", "account"),
    ("I cannot change my profile details", "account"),
    ("How can I update my email address?", "account"),
    ("I need help verifying my account", "account"),
]

RESPONSES = {
    "billing": "Thanks for bringing this to our attention. Please share the order ID and payment reference (do not share your full card number or password). Our billing team can review the transaction and advise on the next step.",
    "technical": "Sorry you're experiencing this issue. Please share your device, app/browser version, and the exact error message. Avoid sending passwords or authentication codes. Our technical team can investigate.",
    "shipping": "We're sorry your delivery is delayed. Please share your order ID so the delivery status can be checked. We will review the tracking information and provide an update.",
    "returns": "We can help with your return or exchange. Please share your order ID and a brief description of the issue. Our team will confirm the applicable return or replacement options.",
    "account": "We can help with your account request. Please share the email address registered with the service (never your password or one-time code), and our support team will guide you through the secure process.",
}

URGENT_TERMS = {"urgent", "immediately", "asap", "emergency", "fraud", "stolen", "locked out", "cannot access"}

POSITIVE_WORDS = {"great", "good", "excellent", "thanks", "thank", "happy", "love", "resolved", "helpful"}
NEGATIVE_WORDS = {"bad", "angry", "terrible", "frustrated", "refund", "broken", "failed", "error", "charged", "late", "delay", "worst", "fraud"}

def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"https?://\S+", " URL ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text)

def _train_model():
    X, y = zip(*TRAINING_DATA)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=3000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit([clean_text(x) for x in X], y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model

def _load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    return _train_model()

def analyze_ticket(message: str) -> dict:
    cleaned = clean_text(message)
    model = _load_model()
    probabilities = model.predict_proba([cleaned])[0]
    classes = model.named_steps["classifier"].classes_
    best_idx = probabilities.argmax()
    category = str(classes[best_idx])
    confidence = float(probabilities[best_idx])

    words = set(cleaned.split())
    positive = len(words & POSITIVE_WORDS)
    negative = len(words & NEGATIVE_WORDS)
    if positive > negative:
        sentiment = "positive"
    elif negative > positive:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    urgent = any(term in cleaned for term in URGENT_TERMS)
    if urgent or (sentiment == "negative" and any(x in cleaned for x in {"fraud", "stolen", "locked"})):
        priority = "high"
    elif sentiment == "negative":
        priority = "medium"
    else:
        priority = "low"

    return {
        "category": category,
        "category_confidence": confidence,
        "sentiment": sentiment,
        "priority": priority,
        "suggested_response": RESPONSES[category],
        "model_name": "TF-IDF + Logistic Regression",
    }
