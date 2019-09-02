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
        H = LH

    h1 = WebsocketClientWorker(id=H.h1_name, port=H.h1_port, host=H.h1_host, hook=hook)
    logger.info("Connected to worker h1.")
    h1 = add_datasets(h1)
    h1.dataset_sizes = {
        "mnist_train": 24754,
        "mnist_test": 10000,
        "eicu_class_train": 4778,
        "eicu_class_test": 5421,
        "eicu_reg_train": 4778,
        "eicu_reg_test": 5421,
    }

    h2 = WebsocketClientWorker(id=H.h2_name, port=H.h2_port, host=H.h2_host, hook=hook)
    logger.info("Connected to worker h2.")
    h2 = add_datasets(h2)
    h2.dataset_sizes = {
        "mnist_train": 17181,
        "mnist_test": 10000,
        "eicu_class_train": 3981,
        "eicu_class_test": 5421,
        "eicu_reg_train": 3981,
        "eicu_reg_test": 5421,
    }

    h3 = WebsocketClientWorker(id=H.h3_name, port=H.h3_port, host=H.h3_host, hook=hook)
    logger.info("Connected to worker h3.")
    h3 = add_datasets(h3)
    h3.dataset_sizes = {
        "mnist_train": 18065,
        "mnist_test": 10000,
        "eicu_class_train": 2387,
        "eicu_class_test": 5421,
        "eicu_reg_train": 2387,
        "eicu_reg_test": 5421,
    }

    return h1, h2, h3


def add_datasets(worker):
    """
    Temp hack to display datasets on the works.

    Args:
        worker:

    Returns:

    """
    worker.datasets = [
        "mnist_train", "mnist_test",
        "eicu_class_train", "eicu_class_test",
        "eicu_reg_train", "eicu_reg_test"
    ]

    worker.dataset_input_dims = {
        "mnist_train": (None, 28, 28),
        "mnist_test": (None, 28, 28),
        "eicu_class_train": (None, 103),
        "eicu_class_test": (None, 103),
        "eicu_reg_train": (None, 103),
        "eicu_reg_test": (None, 103)
    }

    worker.dataset_output_dims = {
        "mnist_train": (None, 10),
        "mnist_test": (None, 10),
        "eicu_class_train": (None, 1),
        "eicu_class_test": (None, 1),
        "eicu_reg_train": (None, 1),
        "eicu_reg_test": (None, 1)
    }

    return worker