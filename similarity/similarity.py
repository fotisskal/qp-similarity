from math import sqrt
import numpy as np
import spacy
import torch
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional
from models.logistic import LogisticRegressionModel
from models.svm import LinearSvcModel
from models.siamese_bert import SiameseBERT
from models.sentence_bert import SentenceBERT
from models.siamese_roberta import SiameseRoBERTa
from transformers import RobertaTokenizer
from fuzzywuzzy import fuzz
from models.nn import NNModel
from utils.text import convert_to_sequence, pad_sequence, preprocess_text
from scipy.sparse import hstack

app = FastAPI(verify_ssl=False)


class ApiModel(BaseModel):
    fwModel: Optional[str] = ''
    text1: Optional[str] = ''
    text2: Optional[str] = ''


@app.get("/")
async def root():
    return {"message": "Welcome to the Text Similarity API"}


@app.post("/cosine-text-similarity")
async def get_conine_text_similarity(request: ApiModel):
    return get_cosine_similarity(request)

@app.post("/cosine-2-text-similarity")
async def get_conine_2_text_similarity(request: ApiModel):
    return get_cosine_2_similarity(request)

# Logistic Regression
@app.post("/lr-model-qp-similarity")
async def lr_model_qp_similarity(request: ApiModel):
    return get_lr_model_qp_similarity(request)

# Linear support vector machine
@app.post("/svm-model-qp-similarity")
async def svm_model_qp_similarity(request: ApiModel):
    return get_svm_model_qp_similarity(request)

# Fuzzywuzzy library
@app.post("/fuzzywuzzy-similarity")
async def fuzzywuzzy_similarity(request: ApiModel):
    return get_fuzzywuzzy_similarity(request)

# Siamese-BERT
@app.post("/siamese-bert-similarity")
async def siamese_bert_similarity(request: ApiModel):
    return get_siamese_bert_similarity(request)

# Siamese-RoBERTa
@app.post("/siamese-roberta-similarity")
async def siamese_roberta_similarity(request: ApiModel):
    return get_siamese_roberta_similarity(request)

# Sentence-BERT
@app.post("/sentence-bert-similarity")
async def siamese_bert_similarity(request: ApiModel):
    return get_sentence_bert_similarity(request)

# Neural network
@app.post("/nn-qp-similarity")
async def nn_qp_similarity(request: ApiModel):
    return get_nn_model_qp_similarity(request)


def get_fuzzywuzzy_similarity(request):
    fw_model = request.fwModel
    similarity_score = 0
    if fw_model == 'simple_ratio':
        print('simple_ratio')
        similarity_score = fuzz.ratio(request.text1, request.text2)
    elif fw_model == 'partial_ratio':
        print('partial_ratio')
        similarity_score = fuzz.partial_ratio(request.text1, request.text2)
    elif fw_model == 'token_sort_ratio':
        print('token_sort_ratio')
        similarity_score = fuzz.token_sort_ratio(request.text1, request.text2)
    else:
        print('token_set_ratio')
        similarity_score = fuzz.token_set_ratio(request.text1, request.text2)
    json_compatible_item_data = jsonable_encoder([
        {
            'similarity': similarity_score
        }
    ])
    return JSONResponse(content=json_compatible_item_data)


def get_siamese_bert_similarity(request):
    model_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

    # Instantiate the SiameseBERT model
    siamese_model = SiameseBERT(model_name)

    # Get the model predictions for the question pairs
    with torch.no_grad():
        outputs = siamese_model(request.text1, request.text2)

    json_compatible_item_data = jsonable_encoder([
        {
            'similarity': outputs.item()
        }
    ])
    return JSONResponse(content=json_compatible_item_data)


def get_siamese_roberta_similarity(request):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = SiameseRoBERTa()

    encoding_1 = tokenizer(request.text1, return_tensors='pt', padding=True, truncation=True)
    encoding_2 = tokenizer(request.text2, return_tensors='pt', padding=True, truncation=True)

    input_ids_1 = encoding_1['input_ids']
    attention_mask_1 = encoding_1['attention_mask']
    input_ids_2 = encoding_2['input_ids']
    attention_mask_2 = encoding_2['attention_mask']

    with torch.no_grad():
        similarity_score = model(input_ids_1=input_ids_1, attention_mask_1=attention_mask_1, 
                       input_ids_2=input_ids_2, attention_mask_2=attention_mask_2)

    json_compatible_item_data = jsonable_encoder([
        {
            'similarity': similarity_score.item()
        }
    ])
    return JSONResponse(content=json_compatible_item_data)


def get_sentence_bert_similarity(request):
    # Initialize two SentenceBERT models
    model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
    sbert_1 = SentenceBERT(model_name)
    sbert_2 = SentenceBERT(model_name)

    # Compute embeddings for two questions
    question1 = "What is the capital of France?"
    question2 = "In which country is Paris located?"
    embedding1 = sbert_1(question1)
    embedding2 = sbert_2(question2)

    # Calculate cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    print(cosine_similarity.item())
    json_compatible_item_data = jsonable_encoder([
        {
            'similarity': cosine_similarity.item()
        }
    ])
    return JSONResponse(content=json_compatible_item_data)


def get_lr_model_qp_similarity(request):
    t1 = preprocess_text(request.text1)
    t2 = preprocess_text(request.text2)
    print(t1)
    print(t2)

    nn = LogisticRegressionModel('questions.csv')

    nn.load_dataset()
    nn.feature_extraction()
    is_trained = nn.is_trained()
    if (is_trained == False):
        nn.split_dataset()
        nn.train()
        nn.evaluate()

    lr_model = nn.get_lr_model()
    vectorizer = nn.get_vectorizer()
    features = vectorizer.fit_transform([t1 + ' ' + t2])

    similarity_score = lr_model.predict_proba(features)[:, 1]
    json_compatible_item_data = jsonable_encoder([
        {
            'similarity': similarity_score[0]
        }
    ])
    return JSONResponse(content=json_compatible_item_data)


def get_svm_model_qp_similarity(request):
    t1 = preprocess_text(request.text1)
    t2 = preprocess_text(request.text2)
    print(t1)
    print(t2)

    nn = LinearSvcModel('quora_questions.csv')

    is_trained = nn.is_trained()
    if (is_trained == False):
        nn.load_dataset()
        nn.feature_extraction()
        nn.split_dataset()
        nn.train()
        nn.evaluate()

    svm_model = nn.get_svm_model()
    vectorizer = nn.get_vectorizer()
    # Vectorize the preprocessed questions
    question1_vec = vectorizer.transform([t1])
    question2_vec = vectorizer.transform([t2])
    question_vec = hstack([question1_vec, question2_vec])
    # X = vectorizer.transform([t1, t2])
    # # Convert the preprocessed text into numerical vectors using n-grams
    # ngram_vectorizer = nn.get_ngram_vectorizer()
    # X_ngram = ngram_vectorizer.transform([t1, t2])
    # question_vec = hstack([X, X_ngram])

    similarity_score = svm_model.predict(question_vec)
    json_compatible_item_data = jsonable_encoder([
        {
            'similarity': similarity_score[0]
        }
    ])
    return JSONResponse(content=json_compatible_item_data)


def get_nn_model_qp_similarity(request):
    nn = NNModel('quora_questions.csv')

    nn.load_dataset()
    nn.feature_extraction()

    word_to_index, embeddings_matrix = nn.generate_embeddings()
    max_sequence_length = nn.get_max_len()

    # preprocess questions
    question1 = preprocess_text(request.text1)
    question2 = preprocess_text(request.text2)

    # convert questions to sequences
    question1_sequence = convert_to_sequence(question1, word_to_index)
    question2_sequence = convert_to_sequence(question2, word_to_index)

    # pad sequences
    question1_sequence = pad_sequence(question1_sequence, max_sequence_length)
    question2_sequence = pad_sequence(question2_sequence, max_sequence_length)

    # reshape sequences
    question1_sequence = np.reshape(question1_sequence, (1, max_sequence_length))
    question2_sequence = np.reshape(question2_sequence, (1, max_sequence_length))

    

    nn.split_dataset()
    nn.create_neural_network((max_sequence_length,), len(word_to_index))
    nn.train()
    nn.evaluate()

    model = nn.get_nn_model()

    prediction = model.predict([question1_sequence, question2_sequence])[0][0]
    json_compatible_item_data = jsonable_encoder([
        {
            'similarity': prediction
        }
    ])
    return JSONResponse(content=json_compatible_item_data)


def get_cosine_similarity(request):
    similarity = compute_similarity(request.text1, request.text2)
    json_compatible_item_data = jsonable_encoder([
        {
            'similarity': similarity
        }
    ])
    return JSONResponse(content=json_compatible_item_data)


def get_cosine_2_similarity(request):
    sentences = [request.text1, request.text2]
    nlp = spacy.load('en_core_web_md')
    embeddings = [nlp(sentence).vector for sentence in sentences]
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    json_compatible_item_data = jsonable_encoder([
        {
            'similarity': similarity
        }

    ])
    return JSONResponse(content=json_compatible_item_data)


def cos_similarity(x,y):
    """ return cosine similarity between two lists """
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = squared_sum(x)*squared_sum(y)
    return round(numerator/float(denominator),3)


def squared_sum(x):
  """ return 3 rounded square rooted value """

  return round(sqrt(sum([a*a for a in x])),3)


def compute_similarity(text1, text2):
    # Preprocess the two texts
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    print(preprocessed_text1)
    print(preprocessed_text2)

    # Create a TfidfVectorizer object to compute the TF-IDF vectors
    vectorizer = TfidfVectorizer()

    # Compute the TF-IDF vectors for the two preprocessed texts
    features = vectorizer.fit_transform([preprocessed_text1, preprocessed_text2])

    # Compute the cosine similarity between the two vectors
    similarity = cosine_similarity(features[0], features[1])[0][0]

    return similarity
