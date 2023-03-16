from pathlib import Path

from config import config


class Log_Service:
    def __init__(self):
        self.logging_file: Path = config['logging']
        self.default_n = config['logging_rows_output']
        pass

    def get_log_rows(self, n=None):
        if n is None:
            n = self.default_n
        with self.logging_file.open("r") as f:
            rows = f.readlines()[-n:]
        return ''.join(rows)
