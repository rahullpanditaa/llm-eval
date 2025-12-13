import string
import json
from nltk.stem import PorterStemmer
from lib.constants import (
    STOPWORDS_FILE_PATH
)

def remove_all_punctuation_lowercase(text: str) -> str:
    tt = str.maketrans("", "", string.punctuation)
    return text.translate(tt).lower()

def tokenize(text: str) -> list[str]:
    return text.lower().split()


def remove_stop_words(tokens: list[str]) -> list[str]:
    with open(STOPWORDS_FILE_PATH, "r") as f:
        words = f.read()
    
    stop_words  = words.splitlines()

    result = []
    for token in tokens:
        if token in stop_words:
            continue
        result.append(token)

    return stem_tokens(result)

def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return list(map(lambda token: stemmer.stem(token), tokens))

def process_text_to_tokens(text: str) -> list[str]:
    tokens = remove_all_punctuation_lowercase(text)
    tokens = tokenize(tokens)
    tokens = remove_stop_words(tokens)
    tokens = stem_tokens(tokens)
    return tokens
