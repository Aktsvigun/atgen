from typing import List
from rouge_score import tokenize
import numpy as np
from atgen.metrics.base_metric import BaseMetric, MetricConfig
from nltk import ngrams
from nltk.stem import porter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import corpus_bleu



class AbstractivenessConfig(MetricConfig):
    aggregate: bool = True

class Abstractiveness(BaseMetric):
    def __init__(self, config: AbstractivenessConfig):
        super().__init__(config)
        
    def _calculate_ngram_overlap(self, summary, text, n=1, use_modified=True):
        summary_ngrams = list(ngrams(summary, n))
        text_ngrams = list(ngrams(text, n))

        if len(summary_ngrams) > 0:
            ngrams_intersection = set(summary_ngrams).intersection(set(text_ngrams))
            if use_modified:
                word_is_part_of_ngram_copied = [
                    any((x in ngram for ngram in ngrams_intersection)) for x in summary
                ]
                return 1 - sum(word_is_part_of_ngram_copied) / len(
                    word_is_part_of_ngram_copied
                )
            else:
                return sum([x not in ngrams_intersection for x in summary_ngrams]) / len(
                    summary_ngrams
                )
        return np.nan


    def compute(self, predictions: List[str], references: List[str], sources: List[str], **kwargs) -> float:
        stemmer = porter.PorterStemmer()
        tokenized_preds = [tokenize.tokenize(x, stemmer) for x in predictions]
        tokenized_texts = [tokenize.tokenize(x, stemmer) for x in sources]
        if references is not None:
            tokenized_refs = [tokenize.tokenize(x, stemmer) for x in references]
        else:
            tokenized_refs = tokenized_preds

        result = {}
        for use_modified in [False, True]:
            for n in range(1, 5):
                pred_ngram_overlaps = []
                label_ngram_overlaps = []
                for pred, label, text in zip(
                    tokenized_preds, tokenized_refs, tokenized_texts
                ):
                    pred_pair_ngram_overlap = self._calculate_ngram_overlap(
                        pred, text, n, use_modified
                    )
                    pred_ngram_overlaps.append(pred_pair_ngram_overlap)
                    if references is not None:
                        label_pair_ngram_overlap = self._calculate_ngram_overlap(
                            label, text, n, use_modified
                        )
                        label_ngram_overlaps.append(label_pair_ngram_overlap)
                key = f"ngram_overlap_{n}" if use_modified else f"novel_ngrams_{n}"

                pred_ngram_overlaps = np.array(pred_ngram_overlaps)
                cond_abs = ~np.isnan(pred_ngram_overlaps)
                result[key + "_abs"] = pred_ngram_overlaps[cond_abs]

                if references is not None:
                    label_ngram_overlaps = np.array(label_ngram_overlaps)
                    cond_rel = cond_abs & ~np.isnan(label_ngram_overlaps)
                    result[key + "_rel"] = (
                        pred_ngram_overlaps[cond_rel] / label_ngram_overlaps[cond_rel]
                    )

        if self.config.aggregate:
            for key, value in result.items():
                result[key] = np.mean(value)

        return {"abstractiveness": result}