from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import pandas as pd
from utils.text import preprocess_text

class LogisticRegressionModel:
    def __init__(self, path):
        self.path = path
        self.lr_model = LogisticRegression(max_iter=500, verbose=1)

    def load_dataset(self):
        self.data = pd.read_csv(self.path)
        self.data['question1'] = self.data['question1'].apply(preprocess_text)
        self.data['question2'] = self.data['question2'].apply(preprocess_text)

    def feature_extraction(self):
        self.vectorizer = TfidfVectorizer()
        self.features = self.vectorizer.fit_transform(self.data['question1'] + ' ' + self.data['question2'])

    def split_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.data['is_duplicate'], test_size=0.2, random_state=42)

    def train(self):
        self.lr_model.fit(self.X_train, self.y_train)
        dump(self.lr_model, 'logreg_model.joblib')

    def evaluate(self):
        y_pred = self.lr_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(accuracy)

    def find_model(self):
        model = load('logreg_model.joblib')
        if model:
            self.lr_model = model
            return True
        return False

    def get_lr_model(self):
        return self.lr_model

    def get_vectorizer(self):
        return self.vectorizer
