from sklearn.model_selection import train_test_split
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import xgboost as xgb
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from os import path
import numpy as np
from utils.text import clean_text

class XGBoostModel:
    def __init__(self, path):
        self.path = path
        self.vectorizer = TfidfVectorizer()
        self.lr_model = self.find_model()

    def load_dataset(self):
        self.data = pd.read_csv(self.path)
        self.data['clean_question1'] = self.data['question1'].apply(clean_text)
        self.data['clean_question2'] = self.data['question2'].apply(clean_text)

    def feature_extraction(self):
        self.vectorizer = TfidfVectorizer()
        self.features = self.vectorizer.fit_transform(self.data['question1'] + ' ' + self.data['question2'])
        # Create the TFIDF-weighted word2vec embeddings
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.data['clean_question1'] + ' ' + self.data['clean_question2'])
        tfidf_feature_names = self.vectorizer.get_feature_names()
        num_features = 300
        question1_vectors = [self.tfidf_weighted_word2vec(sentence, self.wv, self.vectorizer, tfidf_feature_names, num_features) for sentence in self.data['clean_question1']]
        question2_vectors = [self.tfidf_weighted_word2vec(sentence, self.wv, self.vectorizer, tfidf_feature_names, num_features) for sentence in self.data['clean_question2']]
        self.X = np.abs(np.array(question1_vectors) - np.array(question2_vectors))

    def split_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.data['is_duplicate'], test_size=0.2, random_state=42)

    def train(self):
        # create XGBoost DMatrix from the training data
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)

        # set XGBoost parameters
        params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'max_depth': 5}

        # train the model
        num_round = 100
        self.bst = xgb.train(params, dtrain, num_round)
        with open("library/xgb_model.pickle", "wb") as f:
            pickle.dump(self.bst, f)

    def evaluate(self):
        # create XGBoost DMatrix from the testing data
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)

        # make predictions
        y_pred = self.bst.predict(dtest)

        # convert probabilities to binary predictions
        y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]

        # calculate accuracy
        accuracy = sum(y_pred_binary == self.y_test) / len(self.y_test)
        print('Accuracy:', accuracy)

    def find_model(self):
        if path.exists('library/xgb_model.pickle'):
            with open("xgb_model.pickle", "rb") as f:
                self.bst = pickle.load(f)
            print("Model already loaded!")
            self.trained = True
        else:
            self.trained = False

    def get_xgboost_model(self):
        return self.bst
    
    def get_word2vec_model(self):
        # Load the pre-trained word2vec model
        self.wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)    

    def get_vectorizer(self):
        return self.vectorizer

    def is_trained(self):
        return self.trained
    
    def tfidf_weighted_word2vec(sentence, model, tfidf, tfidf_feature_names, num_features):
        # Initialize the vector
        vector = np.zeros(num_features)
        # Get the TFIDF weights
        tfidf_weights = tfidf.transform([sentence]).toarray()[0]
        # Get the word list
        words = sentence.split()
        # Loop over the words
        for i, word in enumerate(words):
            # Check if the word is in the model vocabulary and the TFIDF vocabulary
            if word in model.vocab and word in tfidf_feature_names:
                # Calculate the TFIDF-weighted word vector
                word_vector = model[word]
                weighted_word_vector = word_vector * tfidf_weights[tfidf_feature_names.index(word)]
                # Add the weighted word vector to the sentence vector
                vector = np.add(vector, weighted_word_vector)
        # Return the sentence vector
        return vector
