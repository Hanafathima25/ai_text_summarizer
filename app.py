from flask import Flask, render_template, request,session
import os
from transformers import pipeline
import textstat

app = Flask(__name__)
app.secret_key = os.urandom(24) 
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def split_into_chunks(text, max_tokens=1000):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len((current_chunk + sentence).split()) <= max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    original_text = ""
    original_word_count = 0
    summary_word_count = 0
    compression_ratio = 0
    readability_score = ""
    readability_grade = ""

    if request.method == "POST":
        original_text = request.form["input_text"]
        if original_text.strip():
            original_word_count = len(original_text.split())
            chunks = split_into_chunks(original_text)
            summaries = []

            for chunk in chunks:
                result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
                summaries.append(result[0]['summary_text'])

            summary = " ".join(summaries)
            if "history" not in session:
                session["history"] = []

                session["history"].append({
                    "original": original_text,
                    "summary": summary
                })
            summary_word_count = len(summary.split())
            compression_ratio = round(100 - (summary_word_count / original_word_count * 100), 2)
            readability_score = textstat.flesch_reading_ease(summary)
            readability_grade = textstat.text_standard(summary)


    return render_template("index.html",
                           summary=summary,
                           original_text=original_text,
                           original_word_count=original_word_count,
                           summary_word_count=summary_word_count,
                           compression_ratio=compression_ratio,
                           readability_score=readability_score,
                           readability_grade=readability_grade,
                           history=session.get("history", [])
)

if __name__ == "__main__":
    app.run(debug=True)
