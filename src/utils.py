from pathlib import Path
from typing import List, Optional
import logging

import pandas as pd

from src.config import config


def read_data_file(path, names, **kwargs) -> pd.DataFrame:
    logging.info(f"Try to read file {path}...")
    try:
        return pd.read_csv(path, sep="::", engine="python", header=None, names=names, **kwargs)
    except Exception as e:
        logging.error(str(e))


def read_files(names: List[str], paths: List[Optional[str | Path]]) -> dict[str, pd.DataFrame]:
    data = {}
    for name, param in zip(names, paths):
        arguments = config.data[name]
        if param is not None:
            arguments['path'] = param
        data[name] = read_data_file(**arguments.storage)
    return data


def init_logging():
    logging.basicConfig(
        filename=config.logging,
        encoding='utf-8',
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%d/%b/%Y %H:%M:%S"
    )


def parse_credentials(path):
    with path.open('r') as f:
        text = f.read()
    lines = text.split('\n')
    lines = [line.split(': ') for line in lines]
    result = {line[0]: line[1] for line in lines}
    return result


def save_credentials(dictionary, path):
    lines = [": ".join([key, dictionary[key]]) for key in dictionary]
    text = "\n".join(lines)
    with path.open('w') as f:
        f.write(text)
