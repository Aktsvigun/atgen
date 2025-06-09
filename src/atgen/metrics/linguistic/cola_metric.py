from typing import List, Dict, Union, Optional
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
import nltk

from ..base.base_metric import BaseMetric, MetricConfig

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class ColaMetric(BaseMetric):
    """CoLA (Corpus of Linguistic Acceptability) metric for evaluating grammaticality."""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.checkpoint = (config.checkpoint if config and config.checkpoint else 'Aktsvigun/electra-large-cola')
    
    
    def _initialize_model(self):
        """Initialize the CoLA model if not already initialized."""
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint,
                cache_dir=self.config.cache_dir
            ).to(self.config.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint,
                cache_dir=self.config.cache_dir
            )
    
    def calculate(self, predictions: List[str], references: Optional[List[Union[str, List[str]]]] = None, original_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate CoLA grammaticality scores.
        
        Args:
            predictions: List of predicted texts to evaluate for grammaticality
            references: Not used for CoLA
            original_texts: Not used for CoLA
            
        Returns:
            Dictionary with CoLA scores
        """
        
        self._initialize_model()
        
        # Split texts into sentences
        text_sentences = [nltk.sent_tokenize(text) for text in predictions]
        len_maps = np.cumsum([len(x) for x in text_sentences])
        sentences = [sent for text in text_sentences for sent in text]
        
        # Tokenize sentences
        def tokenize_fn(instance):
            return self.tokenizer(instance["text"], truncation=True)
        
        tokenized_data = Dataset.from_dict({"text": sentences}).map(
            tokenize_fn, remove_columns=["text"], batched=True
        )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataloader = DataLoader(
            tokenized_data, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            collate_fn=data_collator
        )
        
        # Calculate probabilities
        sent_probas = torch.empty(len(sentences), dtype=torch.float32, device=self.config.device)
        probas = torch.empty(len(predictions), dtype=torch.float32, device=self.config.device)
        
        start = 0
        end = self.config.batch_size
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                batch_pred = self.model(**batch)
                batch_probas = 1 / (1 + (-batch_pred.logits[:, 1]).exp())
                sent_probas[start:end].copy_(batch_probas)
                start = end
                end += self.config.batch_size
        
        # Aggregate sentence scores to text scores
        for i, end_idx in enumerate(len_maps):
            start_idx = len_maps[i - 1] if i != 0 else 0
            probas[i].copy_(sent_probas[start_idx:end_idx].mean())
        
        scores = {"cola": probas.cpu().detach().numpy()}
        
        # Aggregate if requested
        if self.config.aggregate:
            scores = {key: float(np.mean(value)) for key, value in scores.items()}
        
        return scores 