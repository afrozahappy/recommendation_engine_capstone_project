#importing the libraries
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing as pre
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics.pairwise import pairwise_distances

 

reviews_df=pd.read_csv("sample30.csv",  encoding = "ISO-8859-1")


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def preprocessing(reviews_new_df):
    reviews_new_df['reviews_data'] = reviews_new_df['reviews_title'] + ' ' + reviews_new_df['reviews_text']
    reviews_new_df=reviews_new_df[['categories','name','reviews_data','user_sentiment']]
    reviews_new_df=reviews_new_df.dropna(how='any',axis=0)
    reviews_new_df['reviews_data'] = reviews_new_df['reviews_data'].str.lower()
    reviews_new_df['reviews_data'] = reviews_new_df['reviews_data'].str.replace('[^A-Za-z\s]+', '')
    nltk.download('stopwords')
    stop = stopwords.words('english')
    reviews_new_df['reviews_data'] = reviews_new_df['reviews_data'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop]))
    p = Counter(" ".join(reviews_new_df['reviews_data']).split()).most_common(10)
    freq_words = [tup[0] for tup in p]
    reviews_new_df['reviews_data'] = reviews_new_df['reviews_data'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in freq_words]))
    p = Counter(" ".join(reviews_new_df['reviews_data']).split())
    sorted_vocabulary = sorted(p.items(), key=lambda x: x[1])
    least_common = [tup[0] for tup in sorted_vocabulary[:10]]
    reviews_new_df['reviews_data'] = reviews_new_df['reviews_data'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in least_common]))
    stemmer = PorterStemmer()
    reviews_new_df['reviews_data'] = reviews_new_df['reviews_data'].str.split().apply(
        lambda x: [stemmer.stem(y) for y in x])
    reviews_new_df['reviews_data'] = reviews_new_df['reviews_data'].apply(lambda x: ' '.join(word for word in x))
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

    reviews_new_df['reviews_data'] = reviews_new_df['reviews_data'].str.split().apply(
        lambda x: [lemmatizer.lemmatize(y) for y in x])
    reviews_new_df['reviews_data'] = reviews_new_df['reviews_data'].apply(lambda x: ' '.join(word for word in x))

    reviews_new_df['reviews_data'] = reviews_new_df['reviews_data'].apply(remove_emoji)

    return reviews_new_df

def tfidf_vectorizer():
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7)
    
    return tfidfconverter

def model_building(reviews_new_df):
    #creating labelEncoder
    le = pre.LabelEncoder()
    # Converting string labels into numbers.
    reviews_new_df['user_sentiment']=le.fit_transform(reviews_new_df['user_sentiment'])

    train, test = train_test_split(reviews_new_df,test_size=0.2,random_state=1)
    train_df = train[['reviews_data', 'user_sentiment']]
    test_df= test['user_sentiment']

    labels_list=test_df.values.tolist()

    dftestdata = test[['reviews_data']]
    testdata_list=dftestdata.values.tolist()
    X = tfidf_vectorizer().fit_transform(reviews_new_df['reviews_data']).toarray()

    y = reviews_new_df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    counter = Counter(y_train)
    # oversampling the train dataset using SMOTE
    smt = SMOTE()
    #X_train, y_train = smt.fit_resample(X_train, y_train)
    X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

    counter = Counter(y_train_sm)
    print('RF')
    text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    text_classifier.fit(X_train_sm, y_train_sm)

    print('model done')
    # Save the model

    # Save the model as a pickle in a file
    joblib.dump(text_classifier, 'models/final_sentiment_analysis_model.pkl')




def recommended_products(ratings):
    dummy_train = ratings.copy()
    dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)
    dummy_train = dummy_train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating',
    aggfunc='mean'
    ).fillna(1)
    df_pivot = ratings.pivot_table(
        index='reviews_username',
        columns='name',
        values='reviews_rating',
        aggfunc='mean'
    )
    print(1)
    mean = np.nanmean(df_pivot, axis=1)
    df_subtracted = (df_pivot.T - mean).T
    user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0
    user_correlation[user_correlation < 0] = 0
    user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
    user_final_rating = np.multiply(user_predicted_ratings, dummy_train)
    joblib.dump(user_final_rating, 'models/user_final_rating.pkl') 
    print('Rec system done')


reviews_new_df=preprocessing(reviews_df)
model_building(reviews_new_df)
user_final_rating=recommended_products(reviews_df)