import logging

import torch
import syft as sy
from syft.workers import WebsocketClientWorker

from neoglia.etl.config import Hospitals as H
from neoglia.etl.config import LocalHospitals as LH

logger = logging.getLogger(__name__)


def connect(local=False):
    """
    Connects to the three hospitals on AWS and returns their WebsocketClientWorkers.
    Args:
        local (bool): Set to true for testing and start_local_workers.py will provide
            three test workers locally.
    Returns:
        tuple(class:`syft.workers.WebsocketClientWorker`): tuple of 3 connected workers.

    """
    hook = sy.TorchHook(torch)
    if local:
        naming = LH
    else:
        naming = H

    h1 = WebsocketClientWorker(
        id=naming.h1_name,
        port=naming.h1_port,
        host=naming.h1_host,
        hook=hook
    )
    logger.info("Connected to worker h1.")

    h2 = WebsocketClientWorker(
        id=naming.h2_name,
        port=naming.h2_port,
        host=naming.h2_host,
        hook=hook
    )
    logger.info("Connected to worker h2.")

    h3 = WebsocketClientWorker(
        id=naming.h3_name,
        port=naming.h3_port,
        host=naming.h3_host,
        hook=hook
    )
    logger.info("Connected to worker h3.")
    return h1, h2, h3
