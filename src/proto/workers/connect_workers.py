import syft as sy
from syft.workers import WebsocketClientWorker
import torch
hook = sy.TorchHook(torch)
h1 = WebsocketClientWorker(id="h1", port=8777, host="ec2-35-178-147-241.eu-west-2.compute.amazonaws.com", hook=hook)
h2 = WebsocketClientWorker(id="h2", port=8777, host="ec2-35-178-10-220.eu-west-2.compute.amazonaws.com", hook=hook)
h3 = WebsocketClientWorker(id="h3", port=8777, host="ec2-3-9-230-244.eu-west-2.compute.amazonaws.com", hook=hook)