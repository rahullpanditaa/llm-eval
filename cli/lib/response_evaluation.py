from lib.generate_answer import AnswerGenerationAndEvaluation
from sentence_transformers import SentenceTransformer
from lib.load_json import InputExtraction
import numpy as np
from lib.utils import process_text_to_tokens


class ResponseEvaluation:
    def __init__(self, conversation_number: int, k: int=5, model_name="all-MiniLM-L6-v2"):
        ie = InputExtraction(conversation_number=conversation_number, k=k)
        self.user_query, _ = ie.extract_info(k=k)
        ga = AnswerGenerationAndEvaluation(conversation_number=conversation_number, k=k)
        answer = ga.generate_answer()
        self.answer = answer["generation"]["answer"]
        self.model = SentenceTransformer(model_name_or_path=model_name)

    def compute_similarity(self):
        question_embedding = self.model.encode([self.user_query])
        answer_embedding = self.model.encode([self.answer])

        sim_score = _cosine_similarity(question_embedding[0], answer_embedding[0])
        return sim_score
    
    def lexical_overlap(self):
        query_tokens = process_text_to_tokens(self.user_query)
        answer_tokens = process_text_to_tokens(self.answer)

        q = set(query_tokens)
        if len(q) == 0:
            return 0.0
        a = set(answer_tokens)
        overlap_count = len(q & a)
        return overlap_count / len(q)

def _cosine_similarity(vec1, vec2):
    """Computes the cosine similrity score between 2 vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)