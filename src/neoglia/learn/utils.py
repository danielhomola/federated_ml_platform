"""
Miscellaneous helper functions for the learning module.
"""

import os
import sys
import yaml
import logging
from time import gmtime, strftime

logger = logging.getLogger(__name__)


def parse_yaml_file(config_file):
    """
    Helper function, parsing all params from a YAML config file.

    Args:
      config_file (str): Path to the config file.
    """
    try:
        with open(config_file) as fd:
            return yaml.load(fd, Loader=yaml.FullLoader)
    except:
        raise IOError("We could not load or parse %s." % config_file)


def setup_logging(logger, loglevel, std_err=True):
    """
    Setup basic stout/file logging.

    Args:
        logger (logger): instantiated logger from runner.
        loglevel (int): minimum loglevel for emitting messages
        std_err (bool): if False logs go to a file not to std err.
    """

    if loglevel is None:
        loglevel = logging.INFO

    handlers = [logging.StreamHandler(sys.stderr)]

    # setup logging folder/file
    if not std_err:
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_name = "%s_%s.log" % (logger.name, strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
        handlers.append(logging.FileHandler(os.path.join(log_dir, log_name)))

    # setup stout and file loggers
    logging.basicConfig(
        level=loglevel,
        format="[%(asctime)s] %(levelname)s:%(name)s:  %(message)s",
        handlers=handlers
    )