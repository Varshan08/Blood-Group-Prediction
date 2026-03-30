import tensorflow as tf
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models

img_size = 100

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(img_size, img_size),
    batch_size=32
)

class_names = train_ds.class_names
print("Classes:", class_names)
model = models.Sequential([
    layers.Rescaling(1./255),
    
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(8, activation='softmax')  # 8 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_ds, epochs=5)

model.save("cnn_model.h5")

print("CNN Model trained ✅")