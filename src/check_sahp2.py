import inspect
from easy_tpp.model.torch_model.torch_sahp import SAHP
print("forward:", inspect.signature(SAHP.forward))
print("loglike_loss:", inspect.signature(SAHP.loglike_loss))