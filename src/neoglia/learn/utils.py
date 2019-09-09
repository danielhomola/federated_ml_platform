"""
Miscellaneous helper functions for the learning module.
"""

import os
import sys
import yaml
import logging
from time import gmtime, strftime

import numpy as np
from sklearn.metrics import auc
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

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


def print_metrics(eval_metrics, ys):
    """
    Helper function to log the metrics that we got back from eval round.
    Args:
        eval_metrics (dict): Dict with scikit learn metrics.
        ys (dict): With y_pred and y_true.
    """
    for metric, metric_value in eval_metrics.items():
        if metric in ["roc_curve", "pr_curve"]:
            x_lab = "FPR" if metric == "roc_curve" else "precision"
            y_lab = "TPR" if metric == "roc_curve" else "recall"
            x_val = metric_value[0] if metric == "roc_curve" else metric_value[1]
            y_val = metric_value[1] if metric == "roc_curve" else metric_value[0]
            metric = "%s AUC" % metric
            try:
                metric_value = auc(x_val, y_val)
            except:
                metric_value = np.nan
            curve_plotter(
                x=x_val,
                y=y_val,
                legend="%s: %.4f" % (metric, metric_value),
                x_lab=x_lab,
                y_lab=y_lab
            )
        elif metric == "confusion_matrix":
            cm_plotter(cm=metric_value, y_pred=ys['y_pred'], y_true=ys['y_true'])
        else:
            logger.info("\t- %s: %.4f" % (metric, metric_value))


def cm_plotter(cm, y_pred, y_true, normalise=False):
    """
    Helper function to plot confusion matrix.
    Args:
        cm (array, shape = [n_classes, n_classes]):
            Output from sklearn.metrics.confusion_matrix.
        y_true (list<int>): Real multi-classes.
        y_pred (list<int>): Predicted multi-classes.
        normalise (bool): Whether to normalise the confusion matrix.
    """
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    classes = ["class %d" % c for c in unique_labels(y_true, y_pred)]

    fig, ax = plt.subplots(1)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title="Confusion matrix",
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


def curve_plotter(x, y, legend, x_lab, y_lab):
    """
    Simple helper to plot the ROC and PR curves at evaluation time.
    Args:
        x (list<float>): quantity to plot on x axis
        y (list<float>): quantity to plot on y axis
        legend (str): legend
        x_lab (str): x label
        y_lab (str): y label

    Returns:

    """
    plt.rcParams["figure.figsize"] = (16, 8)
    plt.figure()
    plt.plot(x, y, color='darkorange', label=legend)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.legend()
    plt.show()
