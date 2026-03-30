import tensorflow as tf

# load old model
model = tf.keras.models.load_model("cnn_model.h5")

# re-save in compatible format
model.save("cnn_model_fixed.h5", save_format="h5")