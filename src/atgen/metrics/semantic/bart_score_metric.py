import traceback
from typing import List, Dict, Union, Optional
import numpy as np
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm

from ..base.base_metric import BaseMetric, MetricConfig


class BARTScorer:
    """Internal BARTScorer implementation."""
    
    def __init__(
        self,
        device="cuda",
        max_length=1024,
        checkpoint="facebook/bart-large-cnn",
        cache_dir: str = "cache",
    ):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
        self.model = BartForConditionalGeneration.from_pretrained(
            checkpoint, cache_dir=cache_dir
        )
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(
            reduction="none", ignore_index=self.model.config.pad_token_id
        )
        self.lsm = nn.LogSoftmax(dim=1)

    def score(self, srcs, tgts, batch_size=4):
        """Score a batch of examples"""
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i : i + batch_size]
            tgt_list = tgts[i : i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    src_tokens = encoded_src["input_ids"].to(self.device)
                    src_mask = encoded_src["attention_mask"].to(self.device)

                    tgt_tokens = encoded_tgt["input_ids"].to(self.device)
                    tgt_mask = encoded_tgt["attention_mask"]
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f"source: {src_list}")
                print(f"target: {tgt_list}")
                exit(0)
        return score_list


class BartScoreMetric(BaseMetric):
    """BARTScore metric for evaluating text generation quality."""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__(config)
        self.scorer = None
        self.checkpoint = getattr(config, 'checkpoint', 'facebook/bart-large-cnn') if config else 'facebook/bart-large-cnn'
    

    def _initialize_scorer(self):
        """Initialize the BARTScorer if not already initialized."""
        if self.scorer is None:
            self.scorer = BARTScorer(
                device=self.config.device,
                checkpoint=self.checkpoint,
                cache_dir=self.config.cache_dir,
            )
    
    def calculate(self, predictions: List[str], references: Optional[List[Union[str, List[str]]]] = None, original_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate BARTScore metrics.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts (can be list of lists for multiple references)  
            original_texts: List of original/source texts
            
        Returns:
            Dictionary with BARTScore results
        """
        
        self._initialize_scorer()
        
        scores = {}
        
        # Calculate source-hypothesis score (BARTScore-sh)
        if original_texts is not None:
            sh_scores = self.scorer.score(original_texts, predictions, batch_size=self.config.batch_size)
            scores["BARTScore-sh"] = np.array(sh_scores)
        
        # Calculate hypothesis-reference score (BARTScore-hr)  
        if references is not None:
            if isinstance(references[0], list):
                # Handle multiple references - take maximum score
                hr_scores = []
                for ref, pred in zip(references, predictions):
                    inst_pred = [pred for _ in range(len(ref))]
                    inst_scores = self.scorer.score(inst_pred, ref, batch_size=self.config.batch_size)
                    hr_scores.append(max(inst_scores))
                scores["BARTScore-hr"] = np.array(hr_scores)
            else:
                hr_scores = self.scorer.score(predictions, references, batch_size=self.config.batch_size)
                scores["BARTScore-hr"] = np.array(hr_scores)
        
        # Aggregate if requested
        if self.config.aggregate:
            scores = {key: float(np.mean(value)) for key, value in scores.items()}
        
        return scores 