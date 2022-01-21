# cat-vs-dog-cnn

**cat-vs-dog classification using cnn**

Dogs vs. Cats dataset provided by Microsoft Research contains 25,000 images of dogs and cats with the labels
link for dataset-https://www.kaggle.com/c/dogs-vs-cats/data
1 = dog
0 = cat
For this project, i used a pre-trained model MobileNetV2 from keras. 
MobileNetV2 is a model that was trained on a large dataset to solve a similar problem to this project, so it will help us to save lots of time on buiding low-level layers and focus on the application.
**CNN**
**Summary of cnn**
 * Provide input image into convolution layer
 * Choose parameters, apply filters with strides, padding if requires. Perform convolution on the image and apply ReLU activation to the matrix.
 * Perform pooling to reduce dimensionality size
 * Add as many convolutional layers until satisfied
 * Flatten the output and feed into a fully connected layer (FC Layer)
 * Output the class using an activation function (Logistic Regression with cost functions) and classifies images.

**how to Build this project**
* Load and preprocess images
* Building CNN model:
 The CNN model contain MobileNetV2, Pooling, fully-connected hidden layer and Output layer.
* Training model:
* Saving the model
* Making prediction from test dataset

**The model has an accuracy of 97.48%**
For each prediction the model gives different proablities prediction.
output:
![Screenshot (516)](https://user-images.githubusercontent.com/90260133/150569796-106f63e4-0a3c-457e-be15-e312c79ef884.png)

