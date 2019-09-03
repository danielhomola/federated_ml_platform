import subprocess

from torchvision import datasets
from torchvision import transforms

mnist_trainset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

call_h1 = [
    "python", "websocket_start.py",
    "--port", "8777",
    "--id", "h1",
    "--host", "localhost"
]
call_h2 = [
    "python", "websocket_start.py",
    "--port", "8778",
    "--id", "h2",
    "--host", "localhost"
]
call_h3 = [
    "python", "websocket_start.py",
    "--port", "8779",
    "--id", "h3",
    "--host", "localhost"
]

subprocess.Popen(call_h1)
subprocess.Popen(call_h2)
subprocess.Popen(call_h3)