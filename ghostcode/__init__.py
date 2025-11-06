from .types import *
from .main import *
from .worker import *
from .utility import *
from .shell import VirtualTerminal 
from .progress_printer import ProgressPrinter
from .internal_testing import *
from .slash_commands import *
from .prompts import *


_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_ghostcode_data(path):
    """Returns PATH preceded by the location of the ghostcode data dir, which is part of the python site-package."""
    return os.path.join(_ROOT, "data", path)
