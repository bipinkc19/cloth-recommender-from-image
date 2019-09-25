from tf_records import DataLoad
import tensorflow as tf
from tensorflow import keras
from datetime import datetime


SUM_OF_ALL_DATASAMPLES = 941489 # Number of augmented images
BATCHSIZE = 1024
EPOCHS = 10

STEPS_PER_EPOCH = SUM_OF_ALL_DATASAMPLES / BATCHSIZE

METHOD = 'try'

# Loggers
logdir = "./tensorboard_logs/" + METHOD + "/" + datetime.now().strftime("%Y-%m-%d//%H-%M-%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

#Get your datatensors
image, label = DataLoad('./tfrecord_files/data_train.tfrecords', 10000, EPOCHS, BATCHSIZE).return_dataset()
val_image, val_label = DataLoad('./tfrecord_files/data_val.tfrecords', 4000, EPOCHS, 4000).return_dataset()

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
                steps_per_epoch=STEPS_PER_EPOCH,
                validation_data=[val_image, val_label],
                validation_steps=STEPS_PER_EPOCH,
                callbacks=[tensorboard_callback])
