import json
from pathlib import Path

SAMPLE_DATA_DIR_PATH = Path(__file__).parent.parent.resolve() / "data" / "samples"

class InputExtraction:
    SAMPLE_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
    sample_chat_jsons = sorted(SAMPLE_DATA_DIR_PATH.glob("sample-chat*.json"))
    sample_vector_jsons = sorted(SAMPLE_DATA_DIR_PATH.glob("sample_context*.json"))
    
   
    def __init__(self, conversation_number: int):
        conversation_path = type(self).sample_chat_jsons[conversation_number]
        context_path = type(self).sample_vector_jsons[conversation_number]
        with open(conversation_path, "r") as f:
            conversation_data = json.load(f)
        with open(context_path, "r") as f:
            context_data = json.load(f)
        
        self.conversation_turns: list[dict] = conversation_data["conversation_turns"]
        self.vector_data: list[dict] = context_data["data"]["vector_data"]

    def _extract_last_user_message(self):
        for turn in self.conversation_turns.reverse():
            if turn["role"] == "User":
                return turn["message"]
    
     # k = number of retrieved contexts to get from sample vector data
    def _extract_context_texts(self, k:int = 5):
        texts = []
        for data in self.vector_data:
            text = data["text"]
            texts.append(text)
        return texts[:k]


        