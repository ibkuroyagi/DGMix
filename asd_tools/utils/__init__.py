from .utils import *  # NOQA
from .pytorch_utils import *  # NOQA


attribute_dict = {
    "bearing": ["loc", "vel", "f-n"],
    "fan": ["m-n", "f-n", "n-lv"],
    "gearbox": ["volt", "wt", "id"],
    "slider": ["vel", "ac", "f-n"],
    "valve": ["v1pat", "pat", "panel", "v2pat"],
    "ToyCar": ["speed", "noise", "mic", "car"],
    "ToyTrain": ["speed", "noise", "mic", "car"],
}
attribute_train_dict = {
    "bearing": [("loc", 8), ("vel", 20), ("f-n", 4)],
    "fan": [("m-n", 4), ("f-n", 4), ("n-lv", 4)],
    "gearbox": [("volt", 13), ("wt", 15), ("id", 12)],
    "slider": [("vel", 17), ("ac", 9), ("f-n", 4)],
    "valve": [("v1pat", 3), ("pat", 8), ("panel", 4), ("v2pat", 2)],
    "ToyCar": [("speed", 5), ("noise", 3), ("mic", 2), ("car", 12)],
    "ToyTrain": [("speed", 5), ("noise", 4), ("mic", 2), ("car", 12)],
}
