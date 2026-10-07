from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential, load_model
from keras.layers import BatchNormalization
from preprocess import process_image
import os
import logging
import numpy as np

def load_saved_model(load_last=True):
    if load_last and os.path.exists(os.path.join('.', 'Models')) and len(os.listdir(os.path.join(".", "Models"))) > 0:
        model_path = os.path.join(".", "Models", f"model{len(os.listdir(os.path.join('.', 'Models')))}.h5")
        logging.info(f'Loading model from {model_path}')
        model = load_model(os.path.join('.', 'Models', f'model{len(os.listdir(os.path.join(".", "Models")))}.h5'))
    else:
        logging.info('No model found. Creating a new one')
        model = Sequential([
            Conv2D(16, 3, activation='relu', input_shape=(145, 145, 3)),
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

    model.save(os.path.join('.', 'Models', f'model{len(os.listdir(os.path.join(".", "Models")))+1}.h5'))

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


def predict(model, image):
    try:
        processed_img = process_image(image, '', '', save_img=False)
        processed_img = processed_img.reshape(-1, 145, 145, 3)
        prediction = model.predict(processed_img)
        return np.where(prediction[0][0] < 0.5, 0, 1)
    except:
        return 2 #1