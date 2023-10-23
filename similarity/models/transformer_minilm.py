from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TransformerMiniLM:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def calculate_similarity(self, question1, question2):
        # Encode the input questions
        embeddings = self.model.encode([question1, question2], convert_to_tensor=True)

        # Calculate cosine similarity between the embeddings
        similarity = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))

        # Return the similarity score
        return similarity[0][0]