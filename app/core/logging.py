import logging


def configure_logging() -> None:
    logger = logging.getLogger()

    if logger.handlers:
        return

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    handler.setFormatter(formatter)

    logger.addHandler(handler)
