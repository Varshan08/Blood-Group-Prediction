import tensorflow as tf

model = tf.keras.models.load_model("cnn_model.h5")

# THIS is the correct way for Keras 3
model.export("cnn_model_tf")