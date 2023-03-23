from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def preprocess_text(text):
    if isinstance(text, str):
        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

        # Stem the remaining words
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        # Lemmatize the remaining words
        # lemmatizer = WordNetLemmatizer()
        # lemmatized_tokens = [lemmatizer.lemmatize(word, pos='v') for word in filtered_tokens]

        # Rejoin the stemmed tokens into a single string
        preprocessed_text = ' '.join(stemmed_tokens)

        return preprocessed_text
    else:
        return ''
