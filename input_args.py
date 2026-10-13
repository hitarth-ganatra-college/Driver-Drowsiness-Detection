import argparse
import sys

def get_input_args():
    parser = argparse.ArgumentParser(description='Driver Drowsiness Detection')

    parser.add_argument('-p', '--preprocess_data', action='store_true', default=False, help='Apply preprocessing to extract facial landmarks')
    parser.add_argument('-t', '--train_model', action='store_true', default=False, help='Train model on data')
    parser.add_argument('--train_dir', type=str, default=None, help='Path to training directory (Dataset2)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train (default: 5)')
    parser.add_argument('--logging_level', type=str, default='debug', help='Set logging level', choices=['debugging', 'info', 'warning'])

    return parser.parse_args()