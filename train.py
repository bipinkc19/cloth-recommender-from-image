from tf_records import DataLoad
import tensorflow as tf
from tensorflow import keras
from datetime import datetime


SUM_OF_ALL_DATASAMPLES = 941489 # Number of augmented images
BATCHSIZE = 32
EPOCHS = 1000

STEPS_PER_EPOCH = SUM_OF_ALL_DATASAMPLES / BATCHSIZE

METHOD = 'try'

# Loggers
logdir = "./content/cloth-recommender-from-image/" + METHOD + "/" + datetime.now().strftime("%Y-%m-%d//%H-%M-%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
tensorboard_callback_collab = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# Get your datatensors
image, label = DataLoad('../drive/My Drive/data_test.tfrecords', 32, EPOCHS, BATCHSIZE).return_dataset()
val_image, val_label = DataLoad('../drive/My Drive/data_test.tfrecords', 32, EPOCHS, BATCHSIZE).return_dataset()

# Combine it with keras
model_input = keras.layers.Input(tensor=image)
inception = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
inception.trainable = False

average_pooling = keras.layers.GlobalAveragePooling2D()

model_output = keras.layers.Dense(46, activation='relu')

train_model = model = keras.Sequential([
    model_input,
    inception,
    average_pooling,
    model_output
])

print(train_model.summary())

# Compile your model
train_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
                    loss='mean_squared_error',
                    metrics=[keras.metrics.categorical_accuracy],
                    target_tensors=[label])

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, verbose=0, mode='min')
mcp_save = keras.callbacks.ModelCheckpoint('./model.hd5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.15, patience=7, verbose=1, min_delta=1e-4, mode='min')

# Train the model
train_model.fit(epochs=EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH,
                validation_data=[val_image, val_label],
                validation_steps=STEPS_PER_EPOCH,
                callbacks=[tensorboard_callback, earlyStopping, mcp_save, reduce_lr_loss, tensorboard_callback_collab])
