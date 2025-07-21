from logging import log
from multiprocessing import reduction
import os
from pathlib import Path
from atgen.metrics.base_metric import BaseMetric, MetricConfig
from typing import List, Optional
from urllib.request import urlretrieve
import numpy as np
from alignscore import AlignScore as AlignScoreModel

class AlignScoreConfig(MetricConfig):
    batch_size: int = 32
    device: str = "cuda"
    cache_dir: str = "cache"


class AlignScore(BaseMetric):
    def __init__(self, config: AlignScoreConfig):
        super().__init__(config)
        self.ALIGNSCORE_CHECKPOINT_PATH = os.getenv(
        "ALIGNSCORE_CHECKPOINT_PATH", '../../cache/AlignScore-base.ckpt',
        )
        

        if not os.path.exists(self.ALIGNSCORE_CHECKPOINT_PATH):
            os.makedirs(os.path.dirname(self.ALIGNSCORE_CHECKPOINT_PATH), exist_ok=True)
            urlretrieve(
                "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt",
                self.ALIGNSCORE_CHECKPOINT_PATH,
            )
            
        self.scorer = AlignScoreModel(
            model="roberta-base",
            batch_size=self.config.batch_size,
            device=self.config.device,
            ckpt_path=self.ALIGNSCORE_CHECKPOINT_PATH,
            evaluation_mode="nli_sp",
        )
        
    def compute(self, predictions: List[str], references: List[str], sources: Optional[List[str]] = None, **kwargs) -> float:
        original_texts = [text if text else " " for text in sources]
        predictions = [text if text else " " for text in predictions]
        references = [text if text else " " for text in references]

        scores_ref = self.scorer.score(contexts=original_texts, claims=predictions)
        if isinstance(references[0], list):
            scores_baseline = []
            for orig_text, refs in zip(original_texts, references):
                inst_baseline_scores = self.scorer.score(
                    contexts=[orig_text] * len(refs), claims=refs
                )
                scores_baseline.append(max(inst_baseline_scores))
        else:
            scores_baseline = self.scorer.score(contexts=original_texts, claims=references)
        scores_rel = np.array(scores_ref) / np.array(scores_baseline)
        
        if self.config.aggregate:
            return {"alignscore": float(np.mean(scores_ref)), "alignscore_rel": float(np.mean(scores_rel))}
        else:
            return {"alignscore": scores_ref, "alignscore_rel": scores_rel}
