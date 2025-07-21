from math import prod
from atgen.metrics.base_metric import BaseMetric, MetricConfig
from typing import List, Optional
import traceback
from typing import List
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration


class BARTScorer:
    def __init__(
        self,
        device="cuda",  # device="cuda:0",
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

    def load(self, path=None):
        """Load model from paraphrase finetuning"""
        if path is None:
            path = "models/bart.pth"
        self.model.load_state_dict(torch.load(path, map_location=self.device))

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

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """Test"""
        src_list = [
            "This is a very good idea. Although simple, but very insightful.",
            "Can I take a look?",
            "Do not trust him, he is a liar.",
        ]

        tgt_list = ["That's stupid.", "What's the problem?", "He is trustworthy."]

        print(self.score(src_list, tgt_list, batch_size))


class BartScoreConfig(MetricConfig):
    device: str = "cuda"
    max_length: int = 1024
    checkpoint: str = "facebook/bart-large-cnn"
    cache_dir: str = "cache"
    batch_size: int = 4

class BartScore(BaseMetric):
    def __init__(self, config: BartScoreConfig):
        super().__init__(config)
        self.scorer = BARTScorer(
            device=self.config.device,
            max_length=self.config.max_length,
            checkpoint=self.config.checkpoint,
            cache_dir=self.config.cache_dir,
        )
        
    def compute(self, predictions: List[str], references: List[str], sources: Optional[List[str]] = None, **kwargs) -> float:
        scores = {}
        if references is not None:
            scores["BARTScore-sh"] = np.array(
                self.scorer.score(references, predictions, batch_size=self.config.batch_size)
        )
        if references is not None:
            if isinstance(references[0], list):
                scores_hr = []
                for ref, pred in zip(references, predictions):
                    inst_pred = [pred for _ in range(len(ref))]
                # Take a maximum within the observation similar to ROUGE
                    inst_score_hr = max(self.scorer.score(inst_pred, ref, batch_size=self.config.batch_size))
                    scores_hr.append(inst_score_hr)
                scores["BARTScore-hr"] = np.array(scores_hr)
        else:
            scores["BARTScore-hr"] = np.array(
                    self.scorer.score(predictions, references, batch_size=self.config.batch_size)
                )

        if self.config.aggregate:
            scores = {key: np.mean(value) for key, value in scores.items()}
        return scores
