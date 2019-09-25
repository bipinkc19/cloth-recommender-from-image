import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2


class TfRecord:
    
    def __init__(self, image_paths, labels, out_path):
        self.image_paths = image_paths
        self.labels = labels
        self.out_path = out_path

    def wrap_int(self, value):
        ''' Wrap the variable in Tensorflow Feature. '''
        return(tf.train.Feature(int64_list=tf.train.Int64List(value=value)))


    def wrap_bytes(self, value):
        ''' Wrap the variable in Tensorflow Feature. '''

        return(tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])))


    def convert_to_tfrecord(self):    
        ''' Convert the images and labels to tfrecord format. '''
        
        print("Converting: " + self.out_path)
    
        # Number of images. Used when printing the progress.
        num_images = len(self.image_paths)

        start = time.time()
        with tf.python_io.TFRecordWriter(self.out_path) as writer:       
            for i, (image, label) in enumerate(zip(self.image_paths, self.labels)):
                if i % 100 == 0:
                    print("progress ", round(i/num_images * 100, 3), '%', round((time.time() - start)/60, 2), ' minutes')

                # img = plt.imread(image)
                
                # # Resize and scale.
                # img = cv2.resize(img, (299, 299))
               
                # # Convert to binary.
                # img_bytes = img.tostring()

                # Just using the raw encoded images to stroe in tfrecords as the decoded byte version is too large
                # img_bytes = open(image, mode='rb').read()

                im = cv2.imread(image)
                im_resize = cv2.resize(im, (299, 299))
                
                # Rescaling the images
                im_resize = im_resize / 255

                is_success, im_buf_arr = cv2.imencode(".jpg", im_resize)
                byte_im = im_buf_arr.tobytes()

                # Create a dictionary to store in tfRecord also wrap in Tensorflow Features.
                data = {
                    'image': self.wrap_bytes(byte_im),
                    'label': self.wrap_int(label)
                }

                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)

                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)

                # Serialize the data.
                serialized = example.SerializeToString()
                        
                # Write the serialized data to the TFRecords file.
                writer.write(serialized)

class DataLoad:

    def __init__(self, file_path, buffer, epochs, batch_size):
        self.buffer = buffer
        self.file_path = file_path
        self.epochs = epochs
        self.batch_size = batch_size

    def parse(self, serialized):
        ''' Decode the binary file tf.record to dictionary. '''
        
        # The dictionary to be stored on.
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            # 46 is the length of array for label.
            'label': tf.FixedLenFeature([46], tf.int64)
        }

        # Converting to example.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                features=features)

        # Decosing the string to unit8.
        image_raw = parsed_example['image']
        # image = tf.decode_raw(image_raw, tf.uint8)
        image = tf.image.decode_jpeg(image_raw)
        
        # Replacing parsed_example dicitonary with float32 value if the image.
        parsed_example['image'] = tf.cast(image, tf.float32)

        return parsed_example['image'], parsed_example['label']


    def return_dataset(self):
        ''' Return the dataset so that we can iterate in it and train model. '''

        dataset = tf.data.TFRecordDataset(self.file_path, num_parallel_reads=16)

        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(self.buffer, self.epochs)
        )
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(self.parse, self.batch_size)
        ) 
        # Prefetch is still not available
        dataset = dataset.apply(
            tf.contrib.data.prefetch_to_device("/gpu:0")
        )

        # Create an iterator
        iterator = dataset.make_one_shot_iterator()
        
        # Create your tf representation of the iterator
        image, label = iterator.get_next()

        # Bring your picture back in shape
        image = tf.reshape(image, [-1, 299, 299, 3])
        
        return image, label

# Prefetch is still not available
# dataset = dataset.apply(
#     tf.contrib.data.prefetch_to_decvice("/gpu:0")
# )
