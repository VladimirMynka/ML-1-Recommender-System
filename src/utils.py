from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import config


def read_data_file(path, names, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="::", engine="python", header=None, names=names, **kwargs)


def read_files(names: List[str], paths: List[Optional[str | Path]]) -> dict[str, pd.DataFrame]:
    data = {}
    for name, param in zip(names, paths):
        arguments = config.data[name]
        if param is not None:
            arguments['path'] = param
        data[name] = read_data_file(**arguments.storage)
        print(data[name].head())
    return data
