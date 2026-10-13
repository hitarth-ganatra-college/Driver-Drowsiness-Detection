import os
os.environ.setdefault('KERAS_BACKEND', 'torch')
import cv2
import logging
import numpy as np
from sklearn.preprocessing import LabelBinarizer
try:
    from keras.preprocessing.image import ImageDataGenerator
except (ImportError, ModuleNotFoundError):
    from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from face_detector_adapter import detect_landmarks, draw_face_landmarks, LEFT_EYE, RIGHT_EYE, MOUTH, ALL_EYE_IDXS

def setup_dirs(categories=None):
    if categories:
        for cat in categories:
            os.makedirs(os.path.join('.', 'Data', 'landmarks', cat), exist_ok=True)
    os.makedirs(os.path.join('.', 'Models'), exist_ok=True)
    os.makedirs(os.path.join('.', 'Logs'), exist_ok=True)

def add_landmarks(name, img_dt, cat, landmarks, lmk_circle_radius=2, save_img=False, black_background=False):
    logging.debug(f'Adding landmarks to {name} image. {"" if save_img else "not "} saving {name} to {os.path.join(".", "Data", "landmarks", cat, str(name))}')

    h, w = img_dt.shape[:2]
    if black_background:
        canvas = np.zeros_like(img_dt)
    else:
        canvas = img_dt.copy()

    for landmark_idx, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        if 0 <= cx < w and 0 <= cy < h:
            color = (0, 255, 0) if landmark_idx in ALL_EYE_IDXS else (255, 255, 255)
            cv2.circle(canvas, (cx, cy), lmk_circle_radius, color, -1)

    if save_img:
        save_dir = os.path.join('.', 'Data', 'landmarks', cat)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, str(name)), canvas)

    return canvas


def process_image(image, category='', name='', face_cas_path=None, save_img=True):
    logging.info(f'Processing {name} in {category}')
    IMG_SIZE = 145
    if image is None:
        return None

    h, w = image.shape[:2]
    landmarks = detect_landmarks(image)
    resized_img = None

    if landmarks:
        drawn_img = add_landmarks(name=name, img_dt=image.copy(), cat=category, landmarks=landmarks, save_img=save_img)
        
        # Compute bounding box from landmarks with 10% padding
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        x_min, x_max = max(0, min(xs)), min(w, max(xs))
        y_min, y_max = max(0, min(ys)), min(h, max(ys))
        pad_x = int((x_max - x_min) * 0.1)
        pad_y = int((y_max - y_min) * 0.1)
        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)
        
        cropped = drawn_img[y_min:y_max, x_min:x_max]
        if cropped.size > 0:
            resized_img = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
        else:
            resized_img = cv2.resize(drawn_img, (IMG_SIZE, IMG_SIZE))
    else:
        # Fallback to Haar Cascade if available
        if face_cas_path and os.path.exists(face_cas_path):
            try:
                face_cascade = cv2.CascadeClassifier(face_cas_path)
                faces = face_cascade.detectMultiScale(image, 1.3, 5)
                if len(faces) > 0:
                    (x, y, fw, fh) = max(faces, key=lambda f: f[2] * f[3])
                    roi = image[y:y+fh, x:x+fw]
                    resized_img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            except Exception as ex:
                logging.debug(f"Haar cascade fallback failed: {ex}")

        if resized_img is None:
            resized_img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    return resized_img

def process_dataset(dir_faces=None, categories=None):
    if categories is None:
        categories = ["Fatigue Subjects", "Active Subjects"]

    if dir_faces is None:
        for candidate in [os.path.join('..', 'Dataset2'), os.path.join('.', 'Dataset2'), r'D:\MCA\Sem_3\DL\Dataset and Model\Dataset2']:
            if os.path.exists(candidate):
                dir_faces = candidate
                break

    logging.info(f'Setting up directories for preprocessing in {dir_faces}')
    setup_dirs(categories)

    imgs_with_landmarks = []
    for category in categories:
        logging.info(f'Processing {category}')
        path_link = os.path.join(dir_faces, category)
        if not os.path.exists(path_link):
            logging.warning(f"Category directory not found: {path_link}")
            continue

        class_num = categories.index(category)
        image_files = os.listdir(path_link)
        logging.info(f"Found {len(image_files)} files in {category}")

        for image in image_files:
            try:
                image_array = cv2.imread(os.path.join(path_link, image))
                if image_array is None:
                    continue
                land_face_array = process_image(image_array, category, image)
                if land_face_array is not None:
                    imgs_with_landmarks.append([land_face_array, class_num])
            except Exception as e:
                logging.warning(f"Could not process {image} in {category}: {e}")

    return imgs_with_landmarks

def setup_training_data(data, test_size=0.2):
    logging.info('Extracting features and labels for training')

    x, y = [], []
    for feature, label in data:
        x.append(feature)
        y.append(label)

    x = np.array(x)
    
    label_bin = LabelBinarizer()
    y = label_bin.fit_transform(y)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=test_size)

    train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
    test_generator = ImageDataGenerator(rescale=1/255)

    train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
    test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)

    return train_generator, test_generator

def load_landmarks(categories):
    landmarks_dir = os.path.join(".", "Data", "landmarks")
    logging.info(f'Loading landmarks from {landmarks_dir}')
    
    IMG_SIZE = 145
    imgs_with_landmarks = []
    for category in categories:
        category_path = os.path.join(landmarks_dir, category)
        if not os.path.exists(category_path):
            logging.warning(f"Category landmark path does not exist: {category_path}")
            continue
        class_num = categories.index(category)

        for image in os.listdir(category_path):
            img_file = os.path.join(category_path, image)
            image_array = cv2.imread(img_file, cv2.IMREAD_COLOR)
            if image_array is not None:
                image_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
                imgs_with_landmarks.append([image_array, class_num])
    
    return imgs_with_landmarks