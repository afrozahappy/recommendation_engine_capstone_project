from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from collections import Counter
#from spellchecker import SpellChecker
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.externals import joblib
from sklearn.metrics.pairwise import pairwise_distances
import colorama
from model import *
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)
model_load = joblib.load("./models/final_sentiment_analysis_model.pkl")
user_final_rating = joblib.load("./models/user_final_rating.pkl")

reviews_df = pd.read_csv("sample30.csv", encoding="ISO-8859-1")


def recommended_products(username):
    d = user_final_rating.loc[username].sort_values(ascending=False)[0:20]
    return list(d.index)




def top_rec_products(output_df):
    positive_review_per={}
    for i in range(output_df.shape[0]):
        if output_df.iloc[i,0] in positive_review_per.keys():
            positive_review_per[output_df.iloc[i,0]][2]=positive_review_per[output_df.iloc[i,0]][2]+1
            if output_df.iloc[i,1]==1:
                positive_review_per[output_df.iloc[i,0]][1] = positive_review_per[output_df.iloc[i,0]][1] + 1
        else:
            positive_review_per[output_df.iloc[i,0]]=[output_df.iloc[i,0],0,1]
            if output_df.iloc[i,1] == 1:
                positive_review_per[output_df.iloc[i,0]][1] = positive_review_per[output_df.iloc[i,0]][1] + 1
    colNames = ['Name', 'Positive_Reviews', 'Total_Reviews']

    count_reviews_df = pd.DataFrame.from_dict(positive_review_per, orient='index')
    count_reviews_df.columns = colNames
    count_reviews_df['positive_percent'] = count_reviews_df.Positive_Reviews / count_reviews_df.Total_Reviews*100
    count_reviews_sorted_df = count_reviews_df.sort_values(by=['positive_percent'], ascending=False)
    return list(count_reviews_sorted_df.iloc[0:5].Name)


def pre_processing(username):
    rec_list=recommended_products(username)
    reviews_new_df = reviews_df[reviews_df.name.isin(rec_list)]
    reviews_new_df=preprocessing(reviews_new_df)
    X = tfidf_vectorizer().transform(reviews_new_df['reviews_data']).toarray()
    output = model_load.predict(X).tolist()
    output_df = pd.DataFrame(output, columns=['user_sentiment'])
    reviews_new_df = reviews_new_df.reset_index(drop=True)
    output_df = output_df.reset_index(drop=True)
    output_new_df = pd.concat([reviews_new_df.name, output_df], axis=1)
    output_new_df=output_new_df.rename(columns = {'name':'item_name'}, inplace = False)
    top_rec_list=top_rec_products(output_new_df)
    return top_rec_list



@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        user = [x for x in request.form.values()]
        rec_list=pre_processing(user)
        final_features = [np.array(user)]
        #output = model_load.predict(final_features).tolist()
        return render_template('index.html', first_prediction=rec_list[0], second_prediction=rec_list[1],
                               third_prediction=rec_list[2], fourth_prediction=rec_list[3], fifth_prediction=rec_list[4])
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run()
