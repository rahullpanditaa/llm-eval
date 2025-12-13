import time
import ollama
from load_json import InputExtraction
from constants import llm_prompt

# based on gpt5.2 pricing
INPUT_COST_PER_TOKEN = 1.750 / 1000000
OUTPUT_COST_PER_TOKEN = 14.000 / 1000000

class AnswerGenerationAndEvaluation:
    def __init__(self, conversation_number: int, k:int = 5):
        ie = InputExtraction(conversation_number=conversation_number)
        self.user_query, self.context_texts = ie.extract_info(k=k)

    def generate_answer(self):
        prompt = llm_prompt(context_texts=self.context_texts, question=self.user_query)
        start_time = time.perf_counter_ns()
        response = ollama.generate(model="mistral:latest", prompt=prompt)
        answer = response.response
        end_time = time.perf_counter_ns()
        
        # in milliseconds
        primary_latency_time = end_time - start_time
        input_tokens = response.prompt_eval_count
        output_tokens = response.eval_count
        model_generated_duration = response.total_duration
        return {
            "generation": {
                "answer": answer,
                "latency_ms": primary_latency_time / 1000000,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                },
                "estimated_cost": (input_tokens * INPUT_COST_PER_TOKEN) + (output_tokens * OUTPUT_COST_PER_TOKEN)
            }
        }