import os
import numpy as np
from keras.preprocessing import image

from scipy.io import loadmat

IMAGE_H = 448
IMAGE_W = 448

def get_labels_from_mat_file(file_path):
    annotations_obj = loadmat(file_path)['annotations']
    annotations = annotations_obj[0]
    labels = np.array([annotation['class'] for annotation in annotations]).flatten()
    return labels


def get_images_numpy_array_and_labels(image_paths, image_labels):
    images_array = []
    # Will store labels for only those images where features could be extracted
    final_labels = []
    for ind, img_path in enumerate(image_paths):
        try:
            img = image.load_img(img_path, target_size=(IMAGE_H, IMAGE_W))
            img = image.img_to_array(img)
            images_array.append(img)
            final_labels.append(image_labels[ind])
        except (OSError, AttributeError, ValueError):
            print('Error in loading image = %s' % img_path)
    images_array = np.array(images_array)
    final_labels = np.array(final_labels)
    return images_array, final_labels


def prep_cars_data():

    train_images_dir = 'cars_data/train_images'
    test_images_dir = 'cars_data/test_images'
    train_annotations_file_path = 'cars_data/devkit/cars_train_annos.mat'
    test_annotations_file_path = 'cars_data/devkit/cars_test_annos.mat'

    train_images_paths = [os.path.join(train_images_dir, img_name) for img_name in os.listdir(train_images_dir)]
    test_images_paths = [os.path.join(test_images_dir, img_name) for img_name in os.listdir(test_images_dir)]

    # Since the labels are coinciding with image name i.e. 0001.jpg will have image label at 1st position
    train_images_paths.sort()
    test_images_paths.sort()

    train_labels = get_labels_from_mat_file(train_annotations_file_path)
    test_labels = get_labels_from_mat_file(test_annotations_file_path)

    X_train, y_train = get_images_numpy_array_and_labels(
        train_images_paths, train_labels)

    X_test, y_test = get_images_numpy_array_and_labels(
        test_images_paths, test_labels)

    return X_train, X_test, y_train, y_test

