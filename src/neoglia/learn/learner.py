"""
Main federated learning module.
"""

import logging
import asyncio
import nest_asyncio

import torch
import syft as sy
from syft.frameworks.torch.federated import utils
from neoglia.learn.utils import setup_logging, print_metrics

logger = logging.getLogger(__name__)
setup_logging(logger, logging.INFO)


class Learner(object):
    """
    Federated learner object.
    """
    def __init__(self, config, model, model_input_dim, loss_fn, workers,
                 evaluate_each_model=True):
        """
        Constructor of the Learner.

        Args:
            config (:class:`neoglia.learn.config.LearnConfig`): LearnConfig object.
            model (:class:`torch.nn.Module`): Neural network defined in Torch.
            model_input_dim (list<int>): Shape of input data to model.
            loss_fn (:class:`torch.jit.ScriptModule`): Loss function as TorchScript.
            workers (tuple(class:`syft.workers.WebsocketClientWorker`): Collection of
                workers whose data to train on.
            evaluate_each_model (bool): If True, all remote models are used to predict
                the test set too in each evaluation round before the federated
                averaging takes place. This allows for tracking remote model performance
        """
        self.config = config
        self.model = model
        self.model_input_dim = model_input_dim
        self.loss_fn = loss_fn
        self.workers = workers
        self.evaluate_each_model = evaluate_each_model
        self.loop = asyncio.get_event_loop()

        # since we're running this from IPython we need to patch the loop
        nest_asyncio.apply(self.loop)

        # setup rest of the environment
        torch.manual_seed(self.config.seed)
        self.serialize_model_send_to_device()

    def serialize_model_send_to_device(self):
        """
        Serializes the model based on the datasets input dimensions and sends it
        to the GPU or CPU.
        """
        # send to device
        use_cuda = self.config.cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(device)

        # serialize model
        dummy_input = torch.zeros(self.model_input_dim, dtype=torch.float)
        self.model.eval()
        self.model = torch.jit.trace(self.model, dummy_input)

    def train_eval(self):
        """
        Main method of the Learner class. Given the config, model, loss and workers,
        this method carries out asynchronous federated averaging on the workers,
        meaning the workers train in parallel and only the evaluation is done in sync.
        """

        return self.loop.run_until_complete(self._train_eval())

    async def _train_eval(self):
        """
        This is the actual main train eval function, but this is not exposed to
        the user.
        """
        for curr_epoch in range(1, self.config.train_epochs + 1):

            # --------------------------------------------------------------------------
            # TRAIN
            # --------------------------------------------------------------------------

            logger.info(
                "Starting epoch %d/%d" % (curr_epoch, self.config.train_epochs)
            )
            results = await asyncio.gather(
                *[
                    self._fit_model_on_worker(worker, curr_epoch)
                    for worker in self.workers
                ]
            )

            # --------------------------------------------------------------------------
            # EVAL
            # --------------------------------------------------------------------------

            test_now = curr_epoch % self.config.fed_after_n_batches == 0

            # first evaluate each remote model separately
            if test_now and self.evaluate_each_model:
                for worker_id, worker_model, _ in results:
                    self._evaluate_model_on_worker(
                        model_identifier=worker_id,
                        worker=self.workers[0],
                        model=worker_model
                    )

            # update the current models and losses to the latest ones
            models = {}
            loss_values = {}
            for worker_id, worker_model, worker_loss in results:
                if worker_model is not None:
                    models[worker_id] = worker_model
                    loss_values[worker_id] = worker_loss

            self.model = utils.federated_avg(models)

            # then, evaluate the averaged model on the test set too
            if test_now:
                self._evaluate_model_on_worker(
                    model_identifier="Federated model",
                    worker=self.workers[0],
                    model=self.model
                )

        if self.config.save_model:
            model_name = "%s_model.pt" % self.config.train_dataset_name
            torch.save(self.model.state_dict(), model_name)

    async def _fit_model_on_worker(self, worker, curr_round):
        """Send the model to the worker and fit the model on the worker's training data.

        Args:
            worker (class:`syft.workers.WebsocketClientWorker`): Remote worker, where
                the model shall be trained.

        Returns:
            A tuple containing:
                * worker_id: Union[int, str], id of the worker.
                * improved model: torch.jit.ScriptModule, model after remote training.
                * loss: Loss on last training batch, torch.tensor.
        """
        train_config = sy.TrainConfig(
            model=self.model,
            loss_fn=self.loss_fn,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            max_nr_batches=self.config.fed_after_n_batches,
            epochs=1,
            optimizer=self.config.optimizer,
            optimizer_args=self.config.optimizer_params,
        )
        train_config.send(worker)

        loss = await worker.async_fit(
            dataset_key=self.config.train_dataset_name,
            return_ids=[0]
        )
        logger.info(
            "Training round: %s, worker: %s, avg_loss: %.4f" %
            (curr_round, worker.id, loss.mean().item())
        )

        model = train_config.model_ptr.get().obj
        return worker.id, model, loss

    def _evaluate_model_on_worker(self, model_identifier, worker, model):
        """
        Method to evaluate models on remote workers. It logs model performance.

        Args:
            model_identifier (str): The model which
            worker (class:`syft.workers.WebsocketClientWorker`): Remote worker, where
                the model shall be evaluated. Note that, all workers have the same test
                set so we always use the first.
            model (:class:`torch.nn.Module`): Neural network defined in Torch.
        """
        model.eval()

        # Create and send train config
        train_config = sy.TrainConfig(
            model=model,
            loss_fn=self.loss_fn,
            batch_size=self.config.test_batch_size,
            optimizer_args=None,
            epochs=1
        )
        train_config.send(worker)
        test_loss, eval_metrics, ys = worker.evaluate(
            dataset_key=self.config.test_dataset_name,
            metrics=self.config.metrics,
            regression=self.config.regression
        )
        logger.info("%s: Test set: Average loss: %.4f" % (model_identifier, test_loss))
        print_metrics(eval_metrics, ys)

