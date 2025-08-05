from atgen.metrics.classic_metrics.bleu import Bleu, BleuConfig
from atgen.metrics.classic_metrics.rouge import Rouge, RougeConfig
from atgen.metrics.classic_metrics.sacrebleu import Sacrebleu, SacrebleuConfig
from atgen.metrics.classic_metrics.exact_match import ExactMatch, ExactMatchConfig
from atgen.metrics.classic_metrics.word_length import WordLength, WordLengthConfig
from atgen.metrics.model_based.bartscore import BartScore, BartScoreConfig
from atgen.metrics.model_based.alignscore import AlignScore, AlignScoreConfig
from atgen.metrics.deep_eval.deepeval_metrics import Deepeval, DeepevalConfig   


METRICS_REGISTRY = {
    "bleu": (Bleu, BleuConfig),
    "rouge": (Rouge, RougeConfig),
    "rouge1": (Rouge, RougeConfig),
    "sacrebleu": (Sacrebleu, SacrebleuConfig),
    "exact_match": (ExactMatch, ExactMatchConfig),
    "word_length": (WordLength, WordLengthConfig),
    "bartscore": (BartScore, BartScoreConfig),
    "alignscore": (AlignScore, AlignScoreConfig),
    "deepeval": (Deepeval, DeepevalConfig),
    
}