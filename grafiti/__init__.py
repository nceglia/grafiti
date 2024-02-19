from . import tools as tl
from . import datasets as ds
from . import plotting as pl
from . import model as ml

import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['tl', 'ds', 'pl', 'ml']})
