import os
_ROOT = os.path.abspath(os.path.dirname(__file__))

def get_ghostcode_data(path: str) -> str:
    """Returns PATH preceded by the location of the ghostcode data dir, which is part of the python site-package."""
    return os.path.join(_ROOT, "data", path)



from .types import *








