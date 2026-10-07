#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
from input_args import get_input_args
from setup_data import setup_kaggle, download_datasets, setup_dirs
from preprocess import process_dataset, setup_training_data, load_landmarks
from model import load_saved_model, train, evaluate

setup_dirs([])

datasets = ['rakibuleceruet/drowsiness-prediction-dataset', 'adinishad/prediction-images']
categories = ["Fatigue Subjects", "Active Subjects"]

def train_model(training_dir, process_data=False, epochs=5):
    if process_data:
        processed_images = process_dataset(training_dir, categories=categories)
    else:
        processed_images = load_landmarks(categories)

    train_generator, test_generator = setup_training_data(processed_images)

    logging.info('Loading model for training')
    model = load_saved_model(load_last=True)

    train(model, train_generator, test_generator, epochs=epochs)

    evaluate(model, test_generator)

if __name__ == '__main__':
    args = get_input_args()
    logging.basicConfig(encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', handlers=[
        logging.FileHandler("./Logs/debug.log"),
        logging.StreamHandler()
    ])

    if args.setup:
        setup_kaggle()

    if args.download_datasets:
        download_datasets(datasets)

    if args.train_model:
        train_model(args.train_dir, args.preprocess_data, 2)


