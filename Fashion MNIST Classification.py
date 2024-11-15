# Import ttthe necessary packages
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
# Reshape the data to include a single channel (grayscale)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# scale data to the range of [0, 1]
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
# One-hot encode the training and testing labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Build the CNN model
model = Sequential()
 # Convolutional layer
    # filters with a kernel size of 5x5
model.add(Conv2D(64, (5, 5),
                      padding="same",
                      activation="relu", 
                      input_shape=(28, 28, 1)))

# MaxPooling layer
# Size with a kernel size of 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5, 5), padding="same",
                      activation="relu"))


model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (5, 5), padding="same", 
                      activation="relu"))

     
model.add(MaxPooling2D(pool_size=(2, 2)))

 # Once the convolutional and pooling 
    # operations are done the layer
    # is flattened and fully connected layers
    # are added
model.add(Flatten())
model.add(Dense(256, activation="relu"))

# Output Layer: Dense layer with 10 neurons (one for each class)
model.add(Dense(10, activation='softmax'))

#Compile the method
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the method
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)

# Evaluate model performance
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

#2 Prediction
# Choose two sample images from the test set
sample_images = X_test[:2]
sample_labels = y_test[:2]

# Make predictions
predictions = model.predict(sample_images)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(sample_labels, axis=1)

# Display the results
for i in range(2):
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_classes[i]}, Actual: {actual_classes[i]}")
    plt.show()


