from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from joblib import dump, load
import pandas as pd
import pandas as pd
from keras.utils import pad_sequences
from keras.layers import Input, Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

class NNModel:
    def __init__(self, path):
        self.path = path

    def load_dataset(self):
        # Load the dataset
        self.data = pd.read_csv(self.path, sep='\t')

    def feature_extraction(self):
        # Tokenize questions
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        encoded_questions = [self.tokenizer.encode(q) for q in self.data['question']]

        # Pad sequences
        self.max_len = max([len(x) for x in encoded_questions])
        self.padded_questions = pad_sequences(encoded_questions, maxlen=self.max_len, padding='post')

    def create_neural_network(self, input_shape, vocab_size):
        # Create the neural network architecture
        embedding_size = 128
        lstm_size = 128
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Embedding layer
        embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=input_shape[0])(inputs)
        
        # LSTM layer
        lstm = LSTM(lstm_size)(embedding)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(lstm)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def split_dataset(self):
        # Split the dataset into training and validation sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.padded_questions, self.data['label'], test_size=0.2, random_state=42)

    def train(self):
        # Train the model on the training set
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=10, batch_size=32, callbacks=[EarlyStopping(patience=3)])
        self.model.save("lstm_model.h5")

    def evaluate(self):
        self.model.evaluate(self.X_test, self.y_test)

    def find_model(self):
        model = load_model('lstm_model.h5')
        if model:
            print("Model already loaded!")
            self.model = model
            return True
        return False

    def get_nn_model(self):
        return self.find_model()

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_max_len(self):
        return self.max_len
    
    def generate_embeddings(self):
        word_to_index = self.tokenizer.get_vocab()
        embeddings_matrix = self.tokenizer.get_input_embeddings().weight.detach().numpy()
        return word_to_index, embeddings_matrix
