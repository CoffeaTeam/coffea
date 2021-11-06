from typing import Optional, List, Any
import logging
import json

try:
    import rich
except ImportError:
    rich = None


class CustomFilter(logging.Filter):
    def __init__(self, modules: Optional[List] = None) -> None:
        self.modules = modules if modules else []
        self.modules.extend(["coffea", "root"])

    def filter(self, record: logging.LogRecord) -> bool:
        base = record.name.split(".")[0]
        return base in self.modules


def setup_logger(
    level: str = "INFO", modules: Optional[List] = None, logfile: Optional[str] = None
) -> logging.Logger:
    """Return a logger. If available, the rich module is used to print nicely.
    In order to avoid too much verbose output from e.g. Dask or numba when setting the level to DEBUG,
    an argument "modules" can be provided in order to select the modules from which logging records are
    picked, filtering the others (coffea is always included).

    Parameters
    ----------
        level: str, optional
            Level of information returned by the logger, can be ither INFO or DEBUG
            Defaults to INFO
        modules: list, optional
            Modules from which we want to get loggin information
        logfile: str, optional
            If specified, the information is dumped not only on stdout but also here
    """
    logger = logging.getLogger()

    # Set up level of information
    possible_levels = ["INFO", "DEBUG"]
    if level not in possible_levels:
        raise ValueError(
            "Passed wrong level for the logger. Allowed levels are: {}".format(
                ", ".join(possible_levels)
            )
        )
    logger.setLevel(getattr(logging, level))

    if rich:
        formatter = logging.Formatter("%(message)s")
    else:
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )

    # Set up filter
    filt = CustomFilter(modules)

    # Set up stream handler (for stdout)
    if rich:
        from rich.logging import RichHandler

        stream_handler = RichHandler(show_time=False, rich_tracebacks=True)
    else:
        stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(filt)
    logger.addHandler(stream_handler)

    # Set up file handler (for logfile)
    if logfile:
        if rich:
            from rich.console import Console

            file_handler = RichHandler(
                show_time=False,
                rich_tracebacks=True,
                console=Console(file=open(logfile, "wt")),
            )
        else:
            file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(filt)
        logger.addHandler(file_handler)

    return logger


def json_str(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=4)
