import sys
import os


# Get the absolute path to the folder *above* this file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add that parent folder to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
from electron_optics.model import *
from electron_optics.utils import *
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--dropout", nargs='+', type=float, default=[0.0], help="Dropout rate(s)")
parser.add_argument("--predictors", nargs='+', default=['predictor1','predictor2','predictor3','predictor4'], help="List of predictors")
args = parser.parse_args()

EPOCH = args.epochs
DROPOUT_LIST = args.dropout
PREDICTORS = args.predictors



with open(current_dir + "/train_vars.json", 'r') as f:
    train_vars = json.load(f)

file_list = [
    os.path.dirname(current_dir)+ "/test_model_data.csv",
    os.path.dirname(current_dir)+ "/parallel_test_model_data.csv",
    os.path.dirname(current_dir)+ "/outlier_target_model_data.csv",
]


for predictor in PREDICTORS:
    predictor_config = train_vars[predictor]
    output_values_start = predictor_config['output_values_start']
    output_values_end = predictor_config['output_values_end']
    n_voltages = predictor_config['n_voltages']
    n_output_values = predictor_config['n_output_values']
    trim_mode = predictor_config['data_trimming']['trim_mode']
    trim_threshold = predictor_config['data_trimming']['threshold']
    leak=predictor_config['leak']
    # plt.ion()
    raw_voltages, raw_outputs = load_data(
        file_list,
        output_values_start=output_values_start,
        output_values_end=output_values_end,
    )
    voltages, outputs, outlier_voltages, outlier_outputs = trim_outliers(
        raw_voltages,
        raw_outputs,
        trim_threshold=trim_threshold,
        trim_mode=trim_mode,
    )

    for dropout in DROPOUT_LIST:

        label_guesser = ElectronOpticsPredictor(
            input_dim=n_voltages, output_dim=n_output_values, leak=leak, dropout=dropout
    )
        print(f"Training {predictor} with dropout={dropout}...")
        label_guesser.train(
            voltages,
            outputs,
            epochs=EPOCH,
            verbose=True,
            checkpoint_name=f"{predictor}_{dropout}.pt",
            make_plot=False,
        )
        
    # plt.ioff()
print("Done!")