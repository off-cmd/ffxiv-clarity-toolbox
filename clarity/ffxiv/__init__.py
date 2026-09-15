from . import exd as exdfile
from . import mtrl as mtrlfile
from . import sqpack, texdecode, texwrite
from . import tex as texfile

try:
    from . import imcfile
except ImportError:
    pass

try:
    from . import mdlstrings
except ImportError:
    pass


def GameData(path):
    return sqpack.GameData(path)


_GD = None


def game(path=None):
    global _GD
    if _GD is None:
        _GD = sqpack.GameData(sqpack.find_game(path))
    return _GD
