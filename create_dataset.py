import pandas as pd
from tf_records import TfRecord
from augment_image import augment_images

# Read the metadata for images and their corresponding labels.
dataset_train = pd.read_csv("./augmented_image/meta_augmented.csv")
dataset_test = pd.read_csv("./test/test.csv")

# Path to store the tfrecord file
out_path_train = "./tfrecord_files/data_train.tfrecords"
out_path_test = "./tfrecord_files/data_test.tfrecords"

# Object of the dataset.

dataset_train = TfRecord(dataset_train['image_id'], dataset_train['labels'], out_path_train)
dataset_test = TfRecord(dataset_test['images'], dataset_test['label'], out_path_test)

# Converts and stores to the out_path.
dataset_train.convert_to_tfrecord()
dataset_test.convert_to_tfrecord()
