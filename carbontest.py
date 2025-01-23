import os
import sys
import torch
import yaml
import argparse
from codecarbon import EmissionsTracker

# Append the parent directory to the path
sys.path.append(os.path.abspath(os.path.join('..')))

import local_datasets
import init_training

# Change working directory if not in /home/mappel/Dynaphos/viseon
if os.getcwd().split("/")[-1] != "viseon":
    os.chdir("/home/mappel/Dynaphos/viseon")

def load_config(yaml_file):
    """Load a YAML configuration file into a dictionary."""
    with open(yaml_file) as file:
        raw_content = yaml.load(file, Loader=yaml.FullLoader)
    return {k: v for params in raw_content.values() for k, v in params.items()}

def run_experiment(config_path):
    """Run a single experiment using the specified configuration file."""
    # Load configuration from yaml file
    cfg = load_config(config_path)

    # Get experiment name from the config filename (without extension and path)
    experiment_name = os.path.basename(config_path).split('.')[0]

    # Load dataset and models based on config
    testset = local_datasets.get_lapa_dataset(cfg, split='test')
    encoder = init_training.get_models(cfg)['encoder']

    # Load model weights
    encoder.load_state_dict(torch.load(cfg['save_path'] + 'checkpoints/final_encoder.pth'))
    encoder.eval()
    encoder.to('cuda')    

    # Run forward pass on a subset of data points
    num_data_points = 2000
    i = 0
    with torch.no_grad():
        for batch in testset:
            example_image = batch['image'].unsqueeze(0).to('cuda')
            with EmissionsTracker(experiment_name, output_file="emissions"+experiment_name+".csv"):
                _ = encoder(example_image)

            i += 1
            if i >= num_data_points:
                break

if __name__ == "__main__":
    # Argument parser for configuration file
    parser = argparse.ArgumentParser(description="Run a single experiment with a specified config file")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to the config file (YAML) for the experiment")
    args = parser.parse_args()

    # Run experiment with specified configuration file
    run_experiment(args.config)
