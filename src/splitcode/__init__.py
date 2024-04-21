import pandas as pd
from collections import OrderedDict
import numpy as np


def count_in_list(ls: list) -> dict:
    count_dict = {}
    for item in ls:
        count_dict[item] = ls.count(item)
    return count_dict


class MultiODict(OrderedDict):
    def __setitem__(self, key, value):
        if isinstance(value, list) and key in self:
            self[key].extend(value)
        else:
            super().__setitem__(key, value)


def text_treatment(serie: pd.Series) -> pd.Series:
    treated_serie = serie.str.normalize('NFKD') \
        .str.encode('ascii', errors='ignore') \
        .str.decode('utf-8') \
        .str.upper() \
        .replace('[^A-Z]', ' ', regex=True) \
        .replace(r'\s+', ' ', regex=True) \
        .str.strip()
    return treated_serie


def idf(n_d: int, N_d: int) -> float:
    return np.log(N_d/n_d)


def tf(n, n_max) -> float:
    return n/n_max



