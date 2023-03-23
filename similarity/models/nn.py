from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import Model
from joblib import dump, load
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from utils.text import preprocess_text

class NNModel:
    def __init__(self, path):
        self.path = path
        self.svm_model = LinearSVC(random_state=42)

    def load_dataset(self):
        # Load the dataset
        self.data = pd.read_csv(self.path, sep='\t')
        # Preprocess the text
        self.data['question1'] = self.data['question1'].apply(preprocess_text)
        self.data['question2'] = self.data['question2'].apply(preprocess_text)

    def feature_extraction(self):
        # Create a tokenizer to convert text into sequences of integers
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.data['question1'] + self.data['question2'])

        # Convert the preprocessed text into sequences of integers
        question1_seq = tokenizer.texts_to_sequences(self.data['question1'])
        question2_seq = tokenizer.texts_to_sequences(df['question2'])

        # Pad the sequences to ensure equal length
        max_len = 50
        question1_seq = pad_sequences(question1_seq, maxlen=max_len, padding='post')
        question2_seq = pad_sequences(question2_seq, maxlen=max_len, padding='post')

    def split_dataset(self):
        # Split the dataset into training and validation sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_combined, self.y, test_size=0.2, random_state=42)

    def train(self):
        # Train a linear support vector machine (SVM) on the training set
        self.svm_model.fit(self.X_train, self.y_train)
        dump(self.svm_model, 'svm_model.joblib')

    def evaluate(self):
        y_pred = self.svm_model.predict(self.X_test)
        # Evaluate the performance of the SVM on the validation set
        print('Accuracy:', accuracy_score(self.y_test, y_pred))
        print('Precision:', precision_score(self.y_test, y_pred))
        print('Recall:', recall_score(self.y_test, y_pred))

    def find_model(self):
        model = load('svm_model.joblib')
        if model:
            self.svm_model = model
            return True
        return False

    def get_svm_model(self):
        return self.svm_model

    def get_vectorizer(self):
        return self.vectorizer
