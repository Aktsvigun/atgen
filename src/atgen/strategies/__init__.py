# from .graph_cut import GraphCutStrategy
from .hadas import HadasStrategy
from .huds import HudsStrategy
from .random_strategy import RandomStrategy
from .te_delfy import TeDelfyStrategy
from .nsp import NSPStrategy
from .bleuvar import BLEUVarStrategy
from .idds import IDDSStrategy
from .dual import DualStrategy


STRATEGIES = {
    "hadas": HadasStrategy,
    "huds": HudsStrategy,
    # "graph_cut": GraphCutStrategy,
    "te_delfy": TeDelfyStrategy,
    "random": RandomStrategy,
    "nsp": NSPStrategy,
    "bleuvar": BLEUVarStrategy,
    "idds": IDDSStrategy,
    "dual": DualStrategy,
    "kek": RandomStrategy,
    "kek2": RandomStrategy,
    "kek3": RandomStrategy,
    "kek4": RandomStrategy,
    "kek5": RandomStrategy,
    "kek6": RandomStrategy,
    "random1": RandomStrategy,
    "random2": RandomStrategy,
}
