from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from joblib import dump, load
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from utils.text import preprocess_text

class LinearSvcModel:
    def __init__(self, path):
        self.path = path
        self.model_exists = False
        self.svm_model = self.find_model()
        self.vectorizer = TfidfVectorizer()
        self.ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))

    def load_dataset(self):
        print('LOAD DATASET')
        self.data = pd.read_csv(self.path, sep='\t')
        self.data['question1'] = self.data['question1'].apply(preprocess_text)
        self.data['question2'] = self.data['question2'].apply(preprocess_text)

    def feature_extraction(self):
        print('FEATURE EXTRACTION')
        # Convert the preprocessed text into numerical vectors using TF-IDF
        X = self.vectorizer.fit_transform(self.data['question1'] + ' ' + self.data['question2'])
        # Convert the preprocessed text into numerical vectors using n-grams
        X_ngram = self.ngram_vectorizer.fit_transform(self.data['question1'] + ' ' + self.data['question2'])
        self.X_combined = hstack([X, X_ngram])
        # Convert the labels into numerical values
        self.y = self.data['is_duplicate'].values.astype(np.int32)

    def split_dataset(self):
        print('SPLIT DATASET')
        # Split the dataset into training and validation sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_combined, self.y, test_size=0.2, random_state=42)

    def train(self):
        print('TRAIN')
        # Train a linear support vector machine (SVM) on the training set
        self.svm_model.fit(self.X_train, self.y_train)
        dump(self.svm_model, 'library/svm_model.joblib')

    def evaluate(self):
        print('EVALUATE')
        y_pred = self.svm_model.predict(self.X_test)
        # Evaluate the performance of the SVM on the validation set
        print('Accuracy: ', accuracy_score(self.y_test, y_pred))
        print('Precision: ', precision_score(self.y_test, y_pred))
        print('Recall: ', recall_score(self.y_test, y_pred))

    def find_model(self):
        model = load('library/svm_model.joblib')
        if model:
            self.trained = True
            return model
        else:
            self.trained = False
            return LinearSVC(random_state=42, tol=1e-5, verbose=1, max_iter=10000)

    def get_svm_model(self):
        return self.svm_model

    def get_vectorizer(self):
        return self.vectorizer
    
    def get_ngram_vectorizer(self):
        return self.ngram_vectorizer
    
    def is_trained(self):
        return self.trained
