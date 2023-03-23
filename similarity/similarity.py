from math import sqrt
import nltk
import spacy
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
from models.logistic import LogisticRegressionModel
from models.svm import LinearSvcModel
from fuzzywuzzy import fuzz
from utils.text import preprocess_text
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

# Neural network
@app.post("/nn-qp-similarity")
async def nn_qp_similarity(request: ApiModel):
    return nn_model_qp_similarity(request)


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


def get_lr_model_qp_similarity(request):
    t1 = preprocess_text(request.text1)
    t2 = preprocess_text(request.text2)
    print(t1)
    print(t2)

    nn = LogisticRegressionModel('questions.csv')

    nn.load_dataset()
    nn.feature_extraction()
    nn.split_dataset()
    nn.train()
    nn.evaluate()

    lr_model = nn.get_lr_model()
    vectorizer = nn.get_vectorizer()
    features = vectorizer.transform([t1 + ' ' + t2])

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

    similarity_score = svm_model.predict(question_vec)
    json_compatible_item_data = jsonable_encoder([
        {
            'similarity': similarity_score[0]
        }
    ])
    return JSONResponse(content=json_compatible_item_data)


def get_nn_model_qp_similarity(request):
    t1 = preprocess_text(request.text1)
    t2 = preprocess_text(request.text2)
    print(t1)
    print(t2)

    nn = LinearSvcModel('quora_questions.csv')

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

    similarity_score = svm_model.predict(question_vec)
    json_compatible_item_data = jsonable_encoder([
        {
            'similarity': similarity_score[0]
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
    similarity = cos_similarity(embeddings[0], embeddings[1])
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
