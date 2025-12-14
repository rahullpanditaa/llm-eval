import nltk
import numpy as np
from sentence_transformers import SentenceTransformer

class CheckHallucination:
    def __init__(self, answer, context_chunks, model_name="all-MiniLM-L6-v2"):
        nltk.download("punkt")
        self.sentences = nltk.sent_tokenize(text=answer)
        self.context_chunks = context_chunks
        self.model = SentenceTransformer(model_name_or_path=model_name)

    def compute_hallucination_score(self):
        support_scores = []
        for sent in self.sentences:
            em = self.model.encode([sent])[0]
            sim_scores_for_current_sent = []
            for ch in self.context_chunks:
                ch_embed = self.model.encode([ch])[0]
                sim_score = _cosine_similarity(em, ch_embed)
                sim_scores_for_current_sent.append(sim_score)
            support_scores.append(max(sim_scores_for_current_sent))

        hallucinated_sentences = 0
        for sc in support_scores:
            if sc < 0.6: #threshold randomly selected
                hallucinated_sentences += 1
        
        hallucination_score = hallucinated_sentences / len(self.sentences)

        if hallucination_score > 0.4:
            note = "Most statements in response lack support from retrieved context"
        elif 0 < hallucination_score <= 0.4:
            note = "Some statements in the answer not supported clearly by retrieved context"
        else:
            note = "All statements in answer supported by retrieved context"
        
        return {
            "hallucination_core": hallucination_score,
            "note": note
        }



def _cosine_similarity(vec1, vec2):
    """Computes the cosine similrity score between 2 vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)