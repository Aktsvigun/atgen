"""Semantic metrics for text evaluation."""

import warnings

# Import with error handling for optional dependencies
__all__ = []

try:
    from .bart_score_metric import BartScoreMetric
    __all__.append("BartScoreMetric")
except ImportError as e:
    warnings.warn(f"BartScore metric not available due to import error: {e}")
    BartScoreMetric = None

try:
    from .alignscore_metric import AlignScoreMetric
    __all__.append("AlignScoreMetric")
except ImportError as e:
    warnings.warn(f"AlignScore metric not available due to import error: {e}")
    AlignScoreMetric = None

try:
    from .sentbert_metric import SentBertMetric
    __all__.append("SentBertMetric")
except ImportError as e:
    warnings.warn(f"SentBert metric not available due to import error: {e}")
    SentBertMetric = None 