## Why TrainConfig doesn't work with PyTorch 1.1+

I spent 5 hours debugging, why in the [advanced parallel mnist example](https://github.com/OpenMined/PySyft/blob/dev/examples/tutorials/advanced/websockets-example-MNIST-parallel/Asynchronous-federated-learning-on-MNIST.ipynb)
we must use PyTorch==1.0.1. Here's what I learned.

PyTorch's JIT capability is amazing. It let's you define a function in Python, and compile that into C++. 
In version 1.0.1 any function or NN module you compile this way with the @torch.jit.script decorator will
return a ScriptModule. This can be quite cleverly [serialised](https://github.com/OpenMined/PySyft/blob/dev/syft/serde/torch_serde.py#L332) and 
[deserializsed](https://github.com/OpenMined/PySyft/blob/dev/syft/serde/torch_serde.py#L332). 
Crucially, this relies on the ScriptModule's save and load methods.

In PyTorch1.1+ the @torch.jit.script decorator will always return a `torch._C.Funtion` object however 
if you want to compile a single function, as explained [here](https://pytorch.org/docs/stable/jit.html).

This object cannot be saved and loaded however as [noted by the community](https://github.com/pytorch/pytorch/issues/20017).

It was also discussed by the [PySyft community](https://github.com/OpenMined/PySyft/issues/2275).

This is ultimately a problem because the `TrainConfig` class requires a loss function that needs to 
be serialised to be sent to the workers for use. This works with PyTorch==1.0.1 as the above 
mentioned serialisation and deserialisation through ScriptModule is fine. 

However, in PyTorch1.1+ this loss function, (regardless of how you try to trick PyTorch, and I have tried really hard) 
will simply become a `torch._C.Function` although can be serialised throug accessing its 
`obj.code` method, but it cannot be deserialised from this. 

Hence,  the proto must run with: porch==1.0.1 and torchvision==0.2.2
