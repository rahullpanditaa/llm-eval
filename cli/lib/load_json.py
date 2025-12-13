import json
from pathlib import Path

SAMPLE_DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data" / "samples"

class InputExtraction:
    # SAMPLE_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
    # sample_chat_jsons = sorted(SAMPLE_DATA_DIR_PATH.glob("sample-chat*.json"))
    # sample_vector_jsons = sorted(SAMPLE_DATA_DIR_PATH.glob("sample_context*.json"))
    # if len(sample_chat_jsons) != len(sample_vector_jsons):
    #     raise ValueError("Error: missing sample conversation or vector data")
    
    def __init__(self, conversation_number: int):
        SAMPLE_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        sample_chat_jsons = sorted(SAMPLE_DATA_DIR_PATH.glob("sample-chat*.json"))
        sample_vector_jsons = sorted(SAMPLE_DATA_DIR_PATH.glob("sample_context*.json"))
        if len(sample_chat_jsons) != len(sample_vector_jsons):
            raise ValueError("Error: missing sample conversation or vector data")
        if conversation_number is None:
            raise ValueError("No conversation selected")
        if conversation_number > len(sample_chat_jsons):
            raise ValueError(f"Number of sample conversations available currently: {len(sample_chat_jsons)}")


        conversation_path = sample_chat_jsons[conversation_number-1]
        context_path = sample_vector_jsons[conversation_number-1]
        with open(conversation_path, "r") as f:
            conversation_data = json.load(f)
        with open(context_path, "r") as f:
            context_data = json.load(f)
        
        self.conversation_turns: list[dict] = conversation_data["conversation_turns"]
        self.vector_data: list[dict] = context_data["data"]["vector_data"]

    def _extract_last_user_message(self) -> str:
        for turn in reversed(self.conversation_turns):
            role = turn.get("role", "").lower()
            if role == "user":
                return turn.get("message", "")
        raise ValueError("No user message found in conversation JSON")
    
     # k = number of retrieved contexts to get from sample vector data
    def _extract_context_texts(self, k:int = 5) -> list[str]:
        texts = []
        for data in self.vector_data:
            text = data.get("text")
            # bug fix - assumed text key always there - threw error
            if text is None:
                continue
            texts.append(text)
        return texts[:k]
    
    def extract_info(self, k:int = 5):
        user_query = self._extract_last_user_message()
        context_texts = self._extract_context_texts(k=k)
        return user_query, context_texts



def extract_info_command(n: int, k:int = 5):
    extractor = InputExtraction(conversation_number=n)
    user_query, context_texts = extractor.extract_info()
    print(f"You have selected conversation number {n}.")
    print("Last user message in selected conversation (abridged to <= 50 chars):")
    print(f"- '{user_query[:50]}'...\n")
    print(f"Context ({k} docs):")
    for i, text in enumerate(context_texts, 1):
        print(f"{i}. {text[:50]}...")

