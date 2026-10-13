import os
os.environ.setdefault('KERAS_BACKEND', 'torch')
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input
from keras.models import Sequential, load_model
from keras.layers import BatchNormalization
from preprocess import process_image
import logging
import numpy as np

def load_saved_model(load_last=True):
    models_dir = os.path.join(".", "Models")
    if load_last and os.path.exists(models_dir):
        saved_files = [f for f in os.listdir(models_dir) if f.endswith('.h5') or f.endswith('.keras')]
        if len(saved_files) > 0:
            saved_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)))
            model_path = os.path.join(models_dir, saved_files[-1])
            logging.info(f'Loading model from {model_path}')
            try:
                model = load_model(model_path)
                model.summary()
                return model
            except Exception as e:
                logging.warning(f"Could not load model from {model_path}: {e}")
    logging.info('No valid saved model found. Creating a new one')
    model = Sequential([
            Input(shape=(145, 145, 3)),
            Conv2D(16, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.1),

            Conv2D(32, 5, activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.1),

            Conv2D(64, 10, activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.1),


            Conv2D(128, 12, activation='relu'),
            BatchNormalization(),

            Flatten(),


            Dense(512, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])

    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    model.summary()
    return model


def train(model, train_generator, test_generator, load_last=True, epochs=5):
    model.fit(train_generator, epochs=epochs, validation_data=test_generator)

    model.save(os.path.join('.', 'Models', f'model{len(os.listdir(os.path.join(".", "Models")))+1}.keras'))

def evaluate(model, test_generator):
    result = model.evaluate(test_generator)
    metric_names = [m.name if hasattr(m, 'name') else str(m) for m in model.metrics] if hasattr(model, 'metrics') and model.metrics else getattr(model, 'metrics_names', ['loss', 'accuracy'])
    if isinstance(result, (list, tuple)):
        metrics = dict(zip(metric_names, result))
        loss = metrics.get('loss', result[0])
        acc = metrics.get('accuracy', result[1] if len(result) > 1 else 'N/A')
    else:
        loss = result
        acc = 'N/A'
    print(f"Model Evaluation Score:")
    print(f"Loss: {loss}")
    print(f"Accuracy: {acc}")


def predict_fatigue_score(model, image):
    """
    Returns (fatigue_prob, binary_label) where fatigue_prob is in [0.0, 1.0].
    Returns (0.0, 0) if inference fails or model is None.
    """
    if model is None or image is None:
        return 0.0, 0
    try:
        processed_img = process_image(image, '', '', save_img=False)
        if processed_img is None:
            return 0.0, 0
        normalized_img = processed_img.astype('float32') / 255.0
        normalized_img = normalized_img.reshape(-1, 145, 145, 3)
        pred = model.predict(normalized_img, verbose=0)
        prob = float(pred[0][0])
        label = 1 if prob >= 0.5 else 0
        return prob, label
    except Exception as e:
        logging.debug(f"CNN prediction exception: {e}")
        return 0.0, 0

def predict(model, image):
    prob, label = predict_fatigue_score(model, image)
    return label