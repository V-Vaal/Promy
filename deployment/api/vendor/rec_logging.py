import functools
import logging
import os
import sys

import paddle.distributed as dist

logger_initialized: dict[str, bool] = {}


@functools.lru_cache()
def get_logger(
    name: str = "ppocr",
    log_file: str | None = None,
    log_level: int = logging.DEBUG,
    log_ranks: str = "0",
) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None and dist.get_rank() == 0:
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_handler = logging.FileHandler(log_file, "a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    rank_ids = [int(rank) for rank in log_ranks.split(",")]
    logger.setLevel(log_level if dist.get_rank() in rank_ids else logging.ERROR)
    logger_initialized[name] = True
    logger.propagate = False
    return logger
