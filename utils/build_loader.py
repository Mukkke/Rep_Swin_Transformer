import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import os

def load_and_preprocess_image(image_path, label, img_size=(224, 224)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0
    return img, label

def create_dataset(batch_size=32, train_data_dir='imagenet/train', img_size=(224, 224), validation_split=0.1):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_dir = os.path.join(current_dir, os.pardir, train_data_dir)
    class_names = os.listdir(train_data_dir)
    class_labels = {class_name: idx for idx, class_name in enumerate(class_names)}
    file_paths = []
    labels = []
    for class_name in class_names:
        class_path = os.path.join(train_data_dir, class_name)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                file_paths.append(file_path)
                labels.append(class_labels[class_name])
                
    labels = tf.one_hot(labels, depth=len(class_names))
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset_size = len(file_paths)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    dataset = dataset.shuffle(dataset_size, reshuffle_each_iteration=False)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset


def create_test_dataset(batch_size=32, test_data_dir='imagenet/val',img_size=(224, 224)):
    class_names = os.listdir(test_data_dir)
    class_labels = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    file_paths = []
    labels = []
    for class_name in class_names:
        class_path = os.path.join(test_data_dir, class_name)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                file_paths.append(file_path)
                labels.append(class_labels[class_name])
    labels = tf.one_hot(labels, depth=len(class_names))

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset









