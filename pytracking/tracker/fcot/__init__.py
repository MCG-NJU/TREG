# from .fcot import FcotTracker
from .fcot_v2 import FcotTracker


def get_tracker_class():
    return FcotTracker