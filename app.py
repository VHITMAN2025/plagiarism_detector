from flask import Flask,render_template,request 
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=['POST'])
def detect():
    input_text = request.form['input_text']

    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)
    prob = model.predict_proba(vectorized_text)[0][1]  # probability of class 1 (plagiarism)
    prob_percent = round(prob * 100, 2)

    if result[0] == 1:
        sentences = [
            "This is a copied sentence from another source.",
            "Another suspicious line found in the text."
        ]
        final_result = f"❌ Plagiarism Detected (Probability: {prob_percent}%)"
    else:
        sentences = []
        final_result = f"✅ No Plagiarism Detected (Probability: {prob_percent}%)"

    return render_template('index.html', result=final_result, sentences=sentences, input_text=input_text)


if __name__ == '__main__':
    app.run(debug=True)