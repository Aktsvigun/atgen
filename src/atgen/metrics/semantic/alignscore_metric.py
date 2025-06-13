import os
from typing import List, Dict, Union, Optional
from urllib.request import urlretrieve
from pathlib import Path
import numpy as np
from alignscore import AlignScore
from ..base.base_metric import BaseMetric, MetricConfig

# Default AlignScore checkpoint path
ALIGNSCORE_CHECKPOINT_PATH = os.getenv(
    "ALIGNSCORE_CHECKPOINT_PATH",
    os.path.join(
        Path(__file__).parents[4],  # Going up to repository root
        "external_metrics/AlignScore/model/AlignScore-base.ckpt",
    ),
)


class AlignScoreMetric(BaseMetric):
    """AlignScore metric for evaluating factual consistency."""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__(config)
        self.scorer = None
        self.checkpoint_path = getattr(config, 'checkpoint_path', ALIGNSCORE_CHECKPOINT_PATH) if config else ALIGNSCORE_CHECKPOINT_PATH
    
    
    def _initialize_scorer(self):
        """Initialize the AlignScore scorer if not already initialized."""
        if self.scorer is None:
            if not os.path.exists(self.checkpoint_path):
                # Download the checkpoint if it doesn't exist
                os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
                urlretrieve(
                    "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt",
                    self.checkpoint_path,
                )
            
            self.scorer = AlignScore(
                model="roberta-base",
                batch_size=self.config.batch_size,
                device=self.config.device,
                ckpt_path=self.checkpoint_path,
                evaluation_mode="nli_sp",
            )
    
    def calculate(self, predictions: List[str], references: Optional[List[Union[str, List[str]]]] = None, original_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate AlignScore metrics.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts (single references only)
            original_texts: List of original/source texts (required)
            
        Returns:
            Dictionary with AlignScore results
        """        
        if original_texts is None:
            raise ValueError("AlignScore requires original texts")
        
        if references is not None and isinstance(references[0], list):
            self.logger.error("AlignScore does not support multiple references. Skipping...")
            return {}
        
        self._initialize_scorer()
        
        # Fix: AlignScore outputs an error if a text is empty, so we need to add some content
        original_texts = [text if text else " " for text in original_texts]
        predictions = [text if text else " " for text in predictions]
        
        scores = {}
        
        # Calculate AlignScore between predictions and original texts
        scores_pred = self.scorer.score(contexts=original_texts, claims=predictions)
        scores["alignscore"] = scores_pred
        
        if references is not None:
            references = [text if text else " " for text in references]
            # Calculate baseline AlignScore between references and original texts
            scores_baseline = self.scorer.score(contexts=original_texts, claims=references)
            # Calculate relative score
            scores_rel = np.array(scores_pred) / np.array(scores_baseline)
            scores["alignscore_rel"] = scores_rel
        
        # Aggregate if requested
        if self.config.aggregate:
            scores = {key: float(np.mean(value)) for key, value in scores.items()}
        
        return scores 