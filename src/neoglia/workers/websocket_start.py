"""
Starts a Websocket server, with 3 datasets:
    MNIST
    eICU mortality classification
    eICU length of stay regression

Each server keeps a subset of these dataset and has a teparate test set too.
"""
import logging

import syft as sy
from syft.workers import WebsocketServerWorker

import torch
import argparse
from torchvision import datasets
from torchvision import transforms
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd


def get_mnist_dataset(keep_labels, training=True):
    """
    Sets up MNIST dataset for training or testing.
    """
    mnist_dataset = datasets.MNIST(
        root="./data",
        train=training,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    # create mnist training
    indices = np.isin(mnist_dataset.targets, keep_labels).astype("uint8")
    logger.info("number of true indices: %s", indices.sum())
    selected_data = (
        torch.masked_select(
            mnist_dataset.data.transpose(0, 2),
            torch.tensor(indices)
            ).view(28, 28, -1).transpose(2, 0)
        )
    logger.info("after selection: %s", selected_data.shape)
    selected_targets = torch.masked_select(
        mnist_dataset.targets,
        torch.tensor(indices)
    )

    return sy.BaseDataset(
        data=selected_data,
        targets=selected_targets,
        transform=mnist_dataset.transform
    )


def get_eicu_dataset(hospitalid, outcome):
    """
    Sets up the eICU dataset for training or testing.
    """
    df_x = pd.read_csv('x.csv')
    to_keep = df_x.hospitalid.values == hospitalid
    df_x.drop('hospitalid', axis=1, inplace=True)
    df_x = df_x[to_keep]
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    x = scaler.fit_transform(df_x.values)

    # load and select outcome
    y = pd.read_csv('y.csv')[outcome][to_keep].values

    return sy.BaseDataset(
        data=torch.from_numpy(x.astype('float32')),
        targets=torch.from_numpy(y.astype('float32'))
    )


def start_websocket_server_worker(id, host, port, hook, verbose, keep_labels=None):
    """
    Helper function for spinning up a websocket server and setting up the local
    datasets: MNIST, eICU for classification and for regression.
    """

    server = WebsocketServerWorker(
        id=id,
        host=host,
        port=port,
        hook=hook,
        verbose=verbose
    )

    # add mnist train & test
    server.add_dataset(
        get_mnist_dataset(keep_labels, training=True),
        key='mnist_train'
    )
    server.add_dataset(
        get_mnist_dataset(list(range(10)), training=False),
        key='mnist_test'
    )

    # add eicu train & test for classification
    id2hospitalid = {
        'h1': 1,
        'h2': 2,
        'h3': 3,
    }
    server.add_dataset(
        get_eicu_dataset(hospitalid=id2hospitalid[id], outcome='hosp_mort'),
        key='eicu_class_train'
    )
    server.add_dataset(
        get_eicu_dataset(hospitalid=4, outcome='hosp_mort'),
        key='eicu_class_test'
    )

    # add eicu train & test for regression
    server.add_dataset(
        get_eicu_dataset(hospitalid=id2hospitalid[id], outcome='icu_los_hours'),
        key='eicu_reg_train'
    )
    server.add_dataset(
        get_eicu_dataset(hospitalid=4, outcome='icu_los_hours'),
        key='eicu_reg_test'
    )

    server.start()
    return server


if __name__ == "__main__":
    # Logging setup
    logger = logging.getLogger("run_websocket_server")
    FORMAT = ("%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d, p:%(process)d) "
              "- %(message)s")
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    # Parse args
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="port number of the websocket server worker, e.g. --port 8777",
    )
    parser.add_argument(
        "--host", type=str,
        default="localhost",
        help="host for the connection"
    )
    parser.add_argument(
        "--id",
        type=str,
        help="name (id) of the websocket server worker, e.g. --id hospital1"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket server worker will be started in verbose mode",
    )

    args = parser.parse_args()

    # define which hospital gets which mnist examples for training
    mnist_keep_labels = {
        "h1": [0, 1, 2, 3],
        "h2": [4, 5, 6],
        "h3": [7, 8, 9],
    }

    # Hook and start server
    hook = sy.TorchHook(torch)
    server = start_websocket_server_worker(
        id=args.id,
        host=args.host,
        port=args.port,
        hook=hook,
        verbose=args.verbose,
        keep_labels=mnist_keep_labels[args.id]
    )
