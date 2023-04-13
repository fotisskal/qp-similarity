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


def convert_to_sequence(text, word_to_index):
    """
    Convert the given text to a sequence of word indices using the provided word-to-index dictionary.

    Args:
        text (str): The text to be converted to a sequence.
        word_to_index (dict): A dictionary mapping words to their corresponding indices.

    Returns:
        A list of integers representing the sequence of word indices for the given text.
    """
    sequence = []
    for word in text.split():
        if word in word_to_index:
            sequence.append(word_to_index[word])
        else:
            sequence.append(word_to_index['<UNK>'])
    return sequence


def pad_sequence(sequence, max_length):
    """
    Pad the given sequence with zeros up to the specified maximum length.

    Args:
        sequence (list): The sequence of integers to be padded.
        max_length (int): The maximum length of the resulting padded sequence.

    Returns:
        A padded sequence of the specified maximum length, represented as a list of integers.
    """
    if len(sequence) >= max_length:
        return sequence[:max_length]
    else:
        num_padding = max_length - len(sequence)
        return sequence + [0] * num_padding
