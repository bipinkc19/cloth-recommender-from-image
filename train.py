from tf_records import DataLoad
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorboard.plugins.beholder import Beholder

# dataset_train = DataLoad('./tfrecord_files/data_train.tfrecords',1, EPOCHS, BATCH_SIZE).return_dataset()


# import tensorflow as tf
# from tensorflow.python import keras as keras
SUM_OF_ALL_DATASAMPLES = 1121753
BATCHSIZE = 128
EPOCHS = 2

STEPS_PER_EPOCH= SUM_OF_ALL_DATASAMPLES / BATCHSIZE
#Get your datatensors
image, label = DataLoad('./tfrecord_files/data_train.tfrecords',1, EPOCHS, BATCHSIZE).return_dataset()

#Combine it with keras
model_input = keras.layers.Input(tensor=image)

#Build your network
model_output = keras.layers.Flatten(input_shape=(-1, 299, 299, 1))(model_input)
model_output = keras.layers.Dense(46, activation='relu')(model_output)

#Create your model
train_model = keras.models.Model(inputs=model_input, outputs=model_output)
print(train_model.summary())
#Compile your model
train_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
                    loss='mean_squared_error',
                    metrics=[keras.metrics.categorical_accuracy],
                    target_tensors=[label])

#Train the model
train_model.fit(epochs=EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH)
