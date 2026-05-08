from pathlib import Path


class SystemService:
    def __init__(self, log_path: Path):

        self.log_path: Path = log_path
