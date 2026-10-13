#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ.setdefault('KERAS_BACKEND', 'torch')

import logging
from input_args import get_input_args
from preprocess import process_dataset, setup_training_data, load_landmarks, setup_dirs
from model import load_saved_model, train, evaluate

setup_dirs([])

datasets = ['Dataset2']
categories = ["Fatigue Subjects", "Active Subjects"]

def train_model(training_dir=None, process_data=False, epochs=5):
    if training_dir is None:
        possible_dirs = [
            os.path.join('..', 'Dataset2'),
            os.path.join('.', 'Dataset2'),
            r'D:\MCA\Sem_3\DL\Dataset and Model\Dataset2'
        ]
        for candidate in possible_dirs:
            if os.path.exists(candidate):
                training_dir = candidate
                break

    logging.info(f"Using training directory: {training_dir}")
    if process_data:
        processed_images = process_dataset(training_dir, categories=categories)
    else:
        processed_images = load_landmarks(categories)
        if len(processed_images) == 0 and training_dir and os.path.exists(training_dir):
            logging.info("No preprocessed landmarks found in Data/landmarks. Running preprocessing on dataset...")
            processed_images = process_dataset(training_dir, categories=categories)

    if len(processed_images) == 0:
        logging.error("No images loaded for training! Ensure the dataset path and categories exist.")
        return

    train_generator, test_generator = setup_training_data(processed_images)

    logging.info('Loading model for training')
    model = load_saved_model(load_last=True)

    train(model, train_generator, test_generator, epochs=epochs)

    evaluate(model, test_generator)

if __name__ == '__main__':
    args = get_input_args()
    log_level = logging.DEBUG if args.logging_level == 'debug' else (logging.INFO if args.logging_level == 'info' else logging.WARNING)
    logging.basicConfig(encoding='utf-8', level=log_level, format='%(asctime)s %(levelname)s %(message)s', handlers=[
        logging.FileHandler("./Logs/debug.log"),
        logging.StreamHandler()
    ])

    if args.train_model:
        train_model(args.train_dir, args.preprocess_data, epochs=args.epochs)


