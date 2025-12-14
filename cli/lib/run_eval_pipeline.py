from lib.load_json import InputExtraction
from lib.generate_answer import AnswerGenerationAndEvaluation
from lib.response_evaluation import ResponseEvaluation
from lib.check_hallucination import CheckHallucination
from pathlib import Path
import json

EVALUATION_RESULTS_PATH = Path(__file__).parent.parent.resolve() / "pipeline_results.json"

def run_evaluation_pipeline(conversation_id: int, k: int=5):
    # input extraction
    user_query, context_texts = InputExtraction(conversation_number=conversation_id).extract_info(k=k)
    _print_extracted_json_inputs(c_id=conversation_id,query=user_query, chunks=context_texts)

    # answer generation and metrics
    answer_and_metrics = AnswerGenerationAndEvaluation(query=user_query, chunks=context_texts).generate_answer()
    _print_generated_answer_and_metrics(answer=answer_and_metrics)
    
    # relevance, completness
    resp_eval = ResponseEvaluation(answer=answer_and_metrics["generation"]["answer"],
                       query=user_query,
                       chunks=context_texts).evaluate_response()
    _print_rel_comp_scores(scores=resp_eval)

    # hallucination
    h_score = CheckHallucination(answer=answer_and_metrics["generation"]["answer"],
                       context_chunks=context_texts).compute_hallucination_score()
    _print_hallucination_score(score=h_score)   

    result = {
        "input": {
            "conversation_id": conversation_id,
            "user_query": user_query,
            "number_retrieved_docs": k
        },
        "generation": {
            "answer": answer_and_metrics["generation"]["answer"],
            "latency_ms": round(answer_and_metrics['generation']['latency_ms'], 2),
            "tokens": answer_and_metrics["generation"]["tokens"],
            "estimated_cost": answer_and_metrics["generation"]["estimated_cost"]
        },
        "evaluation": {
            "relevance": resp_eval["evaluation"]["relevance"],
            "completenss": resp_eval["completeness"]["completeness"],
            "hallucination": h_score
        }
    }
    with open(EVALUATION_RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"- Results saved to '{EVALUATION_RESULTS_PATH.name}'")
    return result




def _print_extracted_json_inputs(c_id:int, query: str, chunks: list[str]):
    print(f"You have selected conversation number {c_id}")
    print("Last user message in selected conversation (abridged to <= 50 chars):")
    print(f"- '{query[:50]}'...\n")
    print(f"Context docs (retrieved k = {len(chunks)}):")
    for i, text in enumerate(chunks, 1):
        print(f"{i}. {text[:50]}...")
    print()

def _print_generated_answer_and_metrics(answer: dict):
    print(f"Generated an answer for the last user message in selected conversation:")
    print(f"- Answer: {answer['generation']['answer']}")
    print(f"- Latency: {answer['generation']['latency_ms']:.5f} ms")
    print(f"- Estimated cost: ${answer['generation']['estimated_cost']:.4f}")
    print(f"- Tokens count: {answer['generation']['tokens']}\n")

def _print_rel_comp_scores(scores: dict):
    print(f"Evaluated LLM response generated for given user query:")
    print(f"- Relevance score: {scores['evaluation']['relevance']['score']}")
    print(f"  - {scores['evaluation']['relevance']['note']}\n")
    print(f"- Completeness score: {scores['evaluation']['completeness']['score']}")
    print(f"  - {scores['evaluation']['completeness']['note']}\n")

def _print_hallucination_score(score: dict):
    print(f"Evaluated hallucination score for generated answer:")
    print(f"- Hallucination score: {score['hallucination_score']}")
    print(f"  {score['note']}\n")
    