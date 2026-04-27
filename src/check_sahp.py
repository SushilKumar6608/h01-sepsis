import inspect
from easy_tpp.model.torch_model.torch_sahp import SAHP
print(inspect.signature(SAHP.forward))
print("---")
# Also check what keys the batch dict contains
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

config = Config.build_from_yaml_file(
    r"D:\Egna projekt\h01-sepsis\configs\sahp_eval_A.yaml",
    experiment_id='SAHP_eval_A'
)
runner = Runner.build_from_config(config)
loader = runner._data_loader.get_loader('dev', shuffle=False)
batch = next(iter(loader))
print("Batch keys:", batch.keys() if hasattr(batch, 'keys') else type(batch))
print("Batch type:", type(batch))
if hasattr(batch, 'keys'):
    for k, v in batch.items():
        print(f"  {k}: {type(v)}, shape={getattr(v, 'shape', 'N/A')}")