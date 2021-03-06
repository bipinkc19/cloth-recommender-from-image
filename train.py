from tf_records import DataLoad
import tensorflow as tf
from tensorflow import keras
from datetime import datetime


SUM_OF_ALL_DATASAMPLES = 941489 # Number of augmented images
BATCHSIZE = 512
EPOCHS = 2

STEPS_PER_EPOCH = SUM_OF_ALL_DATASAMPLES / BATCHSIZE

METHOD = 'try'

# Loggers
logdir = "./tensorboard_logs/" + METHOD + "/" + datetime.now().strftime("%Y-%m-%d//%H-%M-%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Get your datatensors
image, label = DataLoad('../drive/My Drive/data_test.tfrecords', 512, EPOCHS, BATCHSIZE).return_dataset()
val_image, val_label = DataLoad('../drive/My Drive/data_test.tfrecords', 1, 1, 40000).return_dataset()

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

# train_model = keras.models.load_model('../drive/My Drive/model_cloth.hd5')


print(train_model.summary())

# Compile your model
train_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0005),
                    loss='mean_squared_error',
                    metrics=[keras.metrics.categorical_accuracy],
                    target_tensors=[label])

# earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, verbose=0, mode='min')
# mcp_save = keras.callbacks.ModelCheckpoint('../drive/My Drive/model_cloth.hd5', save_best_only=True, monitor='val_loss', mode='min')
# reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.15, patience=7, verbose=1, min_delta=1e-4, mode='min')
save_model_each_epoch = keras.callbacks.ModelCheckpoint('../drive/My Drive/model{epoch:08d}.h5', period=0)

print('*'*20)
# print(tensorboard_callback, earlyStopping, mcp_save, reduce_lr_loss)


# Train the model
train_model.fit(epochs=EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH,
                validation_data=[val_image, val_label],
                validation_steps=1)
train_model.save('../drive/My Drive/final.hdf5')
