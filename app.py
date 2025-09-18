from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

# ==== Load Classifier ====
clf_tokenizer = AutoTokenizer.from_pretrained("./bangla_bert_news_classifier")
clf_model = AutoModelForSequenceClassification.from_pretrained("./bangla_bert_news_classifier")
label_encoder = joblib.load("label_encoder.pkl")
label_map = {i: label for i, label in enumerate(label_encoder.classes_)}

# ==== Load Summarizer ====
sum_tokenizer = AutoTokenizer.from_pretrained("./mt5_bengali_summarizer")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("./mt5_bengali_summarizer")

# ==== Category Prediction ====
def predict_category(text):
    inputs = clf_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = clf_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    category = label_map.get(pred, "unknown")
    return category, outputs.logits

# ==== Softmax for LIME ====
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# ==== LIME Prediction Wrapper (optimized) ====
def lime_predict(texts):
    inputs = clf_tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=128,  # reduced for speed
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = clf_model(**inputs).logits.cpu().numpy()
    return softmax(logits)

# ==== LIME Explanation (optimized) ====
def get_lime_explanation(text, top_k=5):
    explainer = LimeTextExplainer(class_names=list(label_map.values()), split_expression=r'\s+')
    explanation = explainer.explain_instance(
        text,
        lime_predict,
        num_features=top_k,
        num_samples=50  # reduced for speed
    )
    top_words = explanation.as_list()
    total = sum(abs(score) for _, score in top_words) or 1e-6
    token_contributions = [
        f"{word} ({int(100 * abs(score) / total)}%)"
        for word, score in top_words if word.strip()
    ]
    return token_contributions

# ==== Summarization ====
def summarize_text(text):
    inputs = sum_tokenizer([text], max_length=512, truncation=True, return_tensors="pt")
    with torch.no_grad():
        summary_ids = sum_model.generate(
            inputs["input_ids"],
            max_length=100,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    return sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ==== Flask Routes ====
@app.route("/", methods=["GET", "POST"])
def index():
    category, summary, news_text, explanations = None, None, "", None
    if request.method == "POST":
        news_text = request.form["news_text"]
        if news_text.strip():
            category, _ = predict_category(news_text)
            summary = summarize_text(news_text)
            explanations = get_lime_explanation(news_text)
    return render_template(
        "index.html",
        category=category,
        summary=summary,
        news_text=news_text,
        explanations=explanations
    )

if __name__ == "__main__":
    app.run(debug=True)
