"""
train_tpp.py
Trains SAHP via EasyTPP framework on non-sepsis ICU sequences.
"""

import os
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

CONFIG_PATH = r"D:\Egna projekt\h01-sepsis\configs\sahp.yaml"
os.makedirs(r"D:\Egna projekt\h01-sepsis\models\tpp_sahp", exist_ok=True)


def main():
    print("Loading config...")
    config = Config.build_from_yaml_file(
        CONFIG_PATH,
        experiment_id='SAHP_sepsis'
    )

    print("Building runner...")
    runner = Runner.build_from_config(config)

    print("Starting training...")
    runner.run()
    print("Training complete.")


if __name__ == '__main__':
    main()