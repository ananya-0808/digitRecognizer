import os
import tensorflow as tf

# Load and normalize the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train
model.fit(x_train, y_train, epochs=3)

# Evaluate
val_loss, val_acc = model.evaluate(x_test, y_test)
print("loss:", val_loss, "accuracy:", val_acc)

# Save next to this script, no matter where it's run from
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_reader.keras")
model.save(save_path)
print("Saved model to", save_path)