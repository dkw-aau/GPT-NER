"""
Logging
"""
import logging
import os
from logging.handlers import TimedRotatingFileHandler

from pythonjsonlogger import jsonlogger

os.makedirs("./logs", exist_ok=True)

readable_log_handler = logging.StreamHandler(stream=open(1, "w", encoding="utf-8", closefd=False))  # pylint: disable=C0103
readable_log_handler.setFormatter(
    logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s %(pathname)s")
)

file_handler = TimedRotatingFileHandler("./logs/log", when="H")
file_handler.setFormatter(
    jsonlogger.JsonFormatter("%(asctime)-15s %(levelname)-8s %(message)s %(pathname)s", json_ensure_ascii=False)
)
logging.basicConfig(
    level="INFO",
    handlers=[readable_log_handler, file_handler]
)
logging.info("获取logger")


def get_logger(name: str):
    """get sub-loggers"""
    print(f"获取logger: {name}")
    logger = logging.getLogger(name)
    return logger
