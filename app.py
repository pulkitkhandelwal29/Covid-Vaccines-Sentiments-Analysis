from flask import Flask, request, render_template, url_for
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")


# Public Sentiment Analysis
@app.route("/vaccine", methods=["GET", "POST"])
@cross_origin()
def vaccine():
    if request.method == "POST":
        # select vaccine
        selected_vaccine = request.form['vaccines']
        return render_template('index.html', selected_vaccine=selected_vaccine)

    return render_template("index.html")

# Public Sentiment Analysis(zonalreport)
@app.route("/zonalreport", methods=["GET", "POST"])
@cross_origin()
def zonalreport():
    if request.method == "POST":
        return render_template('index.html', selected_vaccine='covishieldvscovaxin')

    return render_template("index.html")     


# Public Sentiment Analysis(vaccinecomparison)
@app.route("/vaccinecomparison", methods=["GET", "POST"])
@cross_origin()
def vaccinecomparison():
    if request.method == "POST":
        return render_template('index.html', selected_vaccine='vaccinesentimentcomparison')

    return render_template("index.html")


# Article Classification
@app.route("/classify", methods=["GET", "POST"])
@cross_origin()
def classify():
    if request.method == "POST":
        model = pickle.load(open("article_classification.pkl", "rb"))
        # article classify
        article_to_classify = request.form['classify_article']
        article_to_classify = [article_to_classify]
        classified_article = model.predict(article_to_classify)
        return render_template('index.html', classified_article=f"{classified_article[0]}")

    return render_template("index.html")




# Article Summarisation
@app.route("/summarise", methods=["GET", "POST"])
@cross_origin()
def summarise():
    if request.method == "POST":
        from transformers import pipeline
        summarizer = pipeline('summarization')
        # article summarise
        article = request.form['long_article']
        summarized_article = summarizer(article, max_length=130, min_length=50, do_sample=False)
        summarized_text = summarized_article[0]['summary_text']

        #Formatting Summarized Text
        summarized_text_list = summarized_text.split(".")
        summarized = []
        for i in range(len(summarized_text_list)):
            capitalized = summarized_text_list[i].lstrip().capitalize()
            summarized.append(capitalized)

        separator = '. '
        final_summarization = separator.join(summarized)
        return render_template('index.html', final_summarization=f"{final_summarization}")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
