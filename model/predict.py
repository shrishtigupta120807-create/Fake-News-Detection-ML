
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "fake_news_model_clean.pkl"

model = joblib.load(MODEL_PATH)

def predict_news(content):
    """
    Predict whether an article is Real or Fake.

    Args:
        content: Combined article title and text.

    Returns:
        Dictionary containing predicted label and model score.
    """

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Article content cannot be empty.")

    prediction = int(model.predict([content])[0])

    probabilities = model.predict_proba([content])[0]
    class_probabilities = dict(zip(model.classes_, probabilities))

    label = "Fake" if prediction == 1 else "Real"

    return {
        "prediction": label,
        "score": round(float(class_probabilities[prediction]) * 100, 2)
    }


if __name__ == "__main__":
    sample = "Your sample news article goes here."

    result = predict_news(sample)
    print(result)
