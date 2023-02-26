import os
import yaml
import logging
import logging.config
import datetime


def set_filename(filename, filedir):
    # with open(r"/content/EMT_BEV/logger_config.yaml", mode="r") as yaml_config:
    with open(
        r"/home/niklas/ETM_BEV/BEVerse/logger_config.yaml", mode="r"
    ) as yaml_config:
        config = yaml.safe_load(yaml_config.read())
        time_stamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        filename = filename + time_stamp + ".log"
        filename = os.path.join(filedir, filename)
        config["handlers"]["file"]["filename"] = filename
    # with open(r"/content/EMT_BEV/logger_config.yaml", mode="w") as yaml_config:
    with open(
        r"/home/niklas/ETM_BEV/BEVerse/logger_config.yaml", mode="w"
    ) as yaml_config:
        yaml.safe_dump(config, yaml_config)
    return config


def setup_custom_logger(
    filename="log",
    filedir="/home/niklas/ETM_BEV/BEVerse/logs/local_logs",  # "/content/drive/MyDrive/logs_thesis/logs"
):
    config = set_filename(filename, filedir)
    logging.config.dictConfig(config)
    logger = logging.getLogger("timelogger")

    return logger
