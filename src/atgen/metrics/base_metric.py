import string
from pydantic import BaseModel
from typing import List, Optional
import re
from abc import ABC, abstractmethod



class MetricConfig(BaseModel):
    aggregate: bool = True


class BaseMetric(ABC):
    def __init__(self, config: MetricConfig):
        self.config = config

    @abstractmethod
    def compute(self, predictions: List[str], references: List[str]) -> float:
        raise NotImplementedError
    
    def _preprocess_text(self, text: str, 
                        do_lowercase: bool = True,
                        do_remove_punctuation: bool = True,
                        do_remove_extra_spaces: bool = True, 
                        do_remove_stopwords: bool = False,
                        stopwords: Optional[list[str]] = None) -> str:
        # Convert to lowercase
        if do_lowercase:
            text = text.lower()
        
        # Remove punctuation
        if do_remove_punctuation:
            # Keep hyphens within words, remove other punctuation
            text = re.sub(r'(?<!\w)-|-(?!\w)', ' ', text)  # Replace standalone hyphens
            translator = str.maketrans('', '', string.punctuation.replace('-', ''))
            text = text.translate(translator)
            text = re.sub(r'(?<!\w)-(?!\w)', '', text)  # Remove remaining standalone hyphens
        
        # Normalize whitespace
        if do_remove_extra_spaces:
            text = ' '.join(text.split())
        
        # Remove stopwords
        if do_remove_stopwords:
            if stopwords is None:
                import nltk
                nltk.download('stopwords')
                stopwords = nltk.corpus.stopwords.words('english')
            words = text.split()
            words = [w for w in words if w not in stopwords]
            text = ' '.join(words)
        
        return text.strip()