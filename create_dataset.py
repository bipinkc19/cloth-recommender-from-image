import pandas as pd
from tf_records import TfRecord
from augment_image import augment_images

def main():
    # Read the metadata for images and their corresponding labels.
    dataset_train = pd.read_csv("./augmented_image/meta_augmented.csv")
    dataset_val = pd.read_csv("./test/val.csv")
    dataset_test = pd.read_csv("./test/test.csv")

    # Path to store the tfrecord file
    out_path_train = "./tfrecord_files/data_train.tfrecords"
    out_path_val = "./tfrecord_files/data_val.tfrecords"
    out_path_test = "./tfrecord_files/data_test.tfrecords"

    # Object of the dataset.
    dataset_train = TfRecord(dataset_train['image_id'], pd.get_dummies(dataset_train['labels'], prefix='class').values, out_path_train)
    dataset_val = TfRecord(dataset_val['images'], pd.get_dummies(dataset_test['label'], prefix='class').values, out_path_test)
    dataset_test = TfRecord(dataset_test['images'], pd.get_dummies(dataset_test['label'], prefix='class').values, out_path_test)

    # Converts and stores to the out_path.
    dataset_train.convert_to_tfrecord()
    dataset_val.convert_to_tfrecord()
    dataset_test.convert_to_tfrecord()

if __name__ == "__main__":
    main()
