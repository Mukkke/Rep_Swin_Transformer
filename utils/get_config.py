import yaml
import os
def get_config(config_file_path = 'config.yaml'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.yaml')
    print(config_path)
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
