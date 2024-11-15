
# Fashion MNIST CNN Classification

## Description
This project demonstrate a Convolutional Neural Network (CNN) using Keras in both Python and R to classify images from the Fashion MNIST dataset. It also makes predictions for two sample images and displays the results.

## Files Included
- Fashion MNIST Classification.py: Python script for running the CNN model.
- README.txt: instructions on how to use the code.


##Requirements
   - Python 3.x
   - R
   - TensorFlow and Keras
   - Numpy
   - Matplotlib## R Version Instructions
  

##Usage
1.Ensure you install the required packages using the following command:
pip install tensorflow numpy matplotlib
2.  Install these packages using the following commands in R:
   install.packages("tensorflow")
   install.packages("keras")
   library(keras)
   library(tensorflow)
   install_tensorflow()
   install_keras()

2. Run the script: 
Fashiona mnist Classification.py
Fashion_mnist Classification.R

3. Expected Output:
   - The script will train the CNN model and display the test accuracy.
   - It will also make predictions on two sample images from the test dataset and show the predicted vs actual labels.

## Notes
- The model architecture includes 2 convolutional layers, 2 max-pooling layers, a flatten layer, a dense layer with 128 neurons, 
  and a final output layer with 10 neurons using softmax activation.
