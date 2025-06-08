from typing import List, Dict, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from ..base.base_metric import BaseMetric, MetricConfig


class SentBertMetric(BaseMetric):
    """SentenceBERT semantic similarity metric."""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.checkpoint = getattr(config, 'checkpoint', 'sentence-transformers/all-mpnet-base-v2') if config else 'sentence-transformers/all-mpnet-base-v2'
    
    
    def _initialize_model(self):
        """Initialize the SentenceBERT model if not already initialized."""
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint, 
                cache_dir=self.config.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.checkpoint, 
                cache_dir=self.config.cache_dir
            ).to(self.config.device)
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging."""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def _compute_similarity(self, source_texts: List[str], ref_texts: List[str]) -> np.ndarray:
        """Compute semantic similarity between source and reference texts."""
        assert len(source_texts) == len(ref_texts)
        
        # Make batch_size an odd number for better batch processing
        batch_size = self.config.batch_size
        if batch_size % 2 == 0:
            batch_size -= 1
        half_batch_size = batch_size // 2
        n_texts = len(source_texts)
        scores = np.empty(n_texts, dtype=np.float32)
        start = 0
        end = 0
        
        while end < n_texts:
            end += half_batch_size
            batch_idx = slice(start, end)
            
            # Tokenize sentences
            encoded_input = self.tokenizer(
                source_texts[batch_idx] + ref_texts[batch_idx],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded_input = {
                key: value.to(self.config.device) for key, value in encoded_input.items()
            }
            
            # Calculate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Perform pooling
            sent_embs = self.mean_pooling(model_output, encoded_input["attention_mask"])
            
            # Normalize embeddings
            sent_embs = F.normalize(sent_embs, p=2, dim=1)
            
            n_source_embs = len(sent_embs) // 2
            scores[batch_idx] = (
                (sent_embs[:n_source_embs] * sent_embs[n_source_embs:])
                .sum(-1)
                .cpu()
                .detach()
                .numpy()
            )
            start = end
        
        return scores
    
    def calculate(self, predictions: List[str], references: Optional[List[Union[str, List[str]]]] = None, original_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate SentenceBERT semantic similarity.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            original_texts: List of original/source texts
            
        Returns:
            Dictionary with semantic similarity scores
        """        
        self._initialize_model()
        
        scores = {}
        
        # Similarity between predictions and references
        if references is not None:
            if isinstance(references[0], list):
                # Handle multiple references - compute average similarity
                ref_scores = []
                for pred, ref_list in zip(predictions, references):
                    pred_list = [pred] * len(ref_list)
                    sim_scores = self._compute_similarity(pred_list, ref_list)
                    ref_scores.append(np.mean(sim_scores))
                scores["sentbert_pred_ref"] = np.array(ref_scores)
            else:
                scores["sentbert_pred_ref"] = self._compute_similarity(predictions, references)
        
        # Similarity between predictions and original texts
        if original_texts is not None:
            scores["sentbert_pred_src"] = self._compute_similarity(predictions, original_texts)
        
        # Aggregate if requested
        if self.config.aggregate:
            scores = {key: float(np.mean(value)) for key, value in scores.items()}
        
        return scores 