
library(keras)
library(tensorflow)

# Load Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# Reshape the data for CNN input and normalize
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))
train_images <- train_images / 255
test_images <- test_images / 255

# One-hot encode the labels
train_labels <- to_categorical(train_labels, 10)
test_labels <- to_categorical(test_labels, 10)

# Build the CNN Model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

# Train the model
history <- model %>% fit(
  train_images, train_labels,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)

# Evaluate the model
score <- model %>% evaluate(test_images, test_labels)
cat('Test accuracy:', score$accuracy, "\n")

# Make predictions on two sample images
predictions <- model %>% predict(test_images[1:2, , , drop = FALSE])
predicted_classes <- apply(predictions, 1, which.max) - 1
actual_classes <- apply(test_labels[1:2, , drop = FALSE], 1, which.max) - 1

# Display the results
for (i in 1:2) {
  image(matrix(test_images[i, , , 1], nrow = 28, ncol = 28)[, 28:1], col = gray.colors(255))
  title(main = paste("Predicted:", predicted_classes[i], "Actual:", actual_classes[i]))
}
