"""
Configuration object and methods for training and evaluation process.
"""
from neoglia.learn.utils import parse_yaml_file


class DotDict(dict):
    """
    Mixin class to make dicts behave a bit like objects.
    Idea from https://stackoverflow.com/a/23689767
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class LearnConfig(DotDict, dict):
    """
    Config dict object, holding all parameters for the training and evaluation.
    """
    def __init__(
            self,
            config_file=None,
            train_dataset_name=None,
            test_dataset_name=None,
            train_batch_size=64,
            test_batch_size=128,
            train_epochs=40,
            fed_after_n_batches=10,
            metrics=["accuracy"],
            lr=0.1,
            cuda=False,
            seed=42,
            save_model=True,
            verbose=True
    ):
        """
        Constructor of the subclassed dict object.

        Args:
            config_file (str): Location of config YAML file. If provided, all
                parameters that are defined within will override the defaults here.
            train_dataset_name (str): Name of the remote dataset to train on.
            test_dataset_name (str): Name of the remote dataset to test on.
            train_batch_size (int): Batch size for training.
            test_batch_size (int): Batch size for evaluation.
            train_epochs (int): Number of epochs performed altogether for training on
                remote workers.
            fed_after_n_batches (int): Number of training epochs performed on each
                remote worker before averaging global model.
            metrics (tuple<str>): Metrics to use for evaluation of the model. Use any
                of: accuracy, precision, recall, mse, mae.
            lr (float): Learning rate for the optimizer.
            cuda (bool): Whether the remote workers have GPUs and CUDA enabled.
            seed (int): Seed for reproducibility.
            save_model (bool): Whether to save the global model. If yes, it is
                saved where the python interpreter is running.
            verbose (bool): Verbosity - false: not entirely silent, but quite minimal.
        """
        super(LearnConfig, self).__init__()

        self.__setitem__('config_file', config_file)
        self.__setitem__('train_dataset_name', train_dataset_name)
        self.__setitem__('test_dataset_name', test_dataset_name)
        self.__setitem__('train_batch_size', train_batch_size)
        self.__setitem__('test_batch_size', test_batch_size)
        self.__setitem__('train_epochs', train_epochs)
        self.__setitem__('fed_after_n_batches', fed_after_n_batches)
        self.__setitem__('metrics', metrics)
        self.__setitem__('lr', lr)
        self.__setitem__('cuda', cuda)
        self.__setitem__('seed', seed)
        self.__setitem__('save_model', save_model)
        self.__setitem__('verbose', verbose)

        # set class-reg variable
        self.__setitem__('regression', False)

        # check that we don't have incompatible metrics
        if "mse" in self.metrics or "mae" in self.metrics:
            error = "%s cannot be used with mse and mae!"
            for metric in ["accuracy", "precision", "recall"]:
                assert metric not in self.metrics, error % metric
            self.__setitem__('regression', True)

        # if there was a config file provided, override the params that are define in it
        if config_file is not None:
            self.update(parse_yaml_file(config_file))
