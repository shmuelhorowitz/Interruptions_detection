from urllib.parse import urlparse
import pickle
import time

import numpy as np
import pandas as pd

def parse_time_window(time_window):
    """
    receive time window with format m:s-m:s or s:s and return first time in seconds and last time in second.
    for example 50-1:10 return (50,70)
    """
    min2sec_changer = lambda t: int(t.split(":")[0]) * 60 + int(t.split(":")[1]) if ":" in t else int(t)
    return min2sec_changer(time_window.split("-")[0]), min2sec_changer(time_window.split("-")[1])


def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


class DotConfig:
    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, k):
        v = self._cfg[k]
        if isinstance(v, dict):
            return DotConfig(v)
        return v