import logging
import logging.config
from pathlib import Path

import yaml


def setup_logging(
    config_path: str = "config/logging.yaml",
    logs_dir: str = "logs",
) -> None:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
