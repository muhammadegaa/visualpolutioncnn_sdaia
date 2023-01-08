import tensorflow as tf

# Load the dataset and split it into training, validation, and test sets
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset()

# Normalize the data by scaling the pixel values to a range between 0 and 1
x_train, x_val, x_test = x_train / 255, x_val / 255, x_test / 255

# Build the CNN model using the functional API
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model with an optimizer and a loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training set and evaluate it on the validation set
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))

# Plot the training and validation accuracy and loss
plot_history(history)

# Test the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, return_dict=True)
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')
