# Arabic Handwritten Characters Classification Using Logistic Regression, SVM, and Neural Networks

## Abstract

The Arabic Letter dataset (with 16,800 32x32 RGB images) was used in this research project. The dataset was separated into training set (13,440 images) and testing set (3,360 images). Three machine learning algorithms, namely Support Vector Machine (SVM), Logistic Regression and Convolutional Neural Network (CNN) were trained on the data. Linear, radial basis function (RBF) and sigmoid function kernel were used when training the data with SVMs, with the penalty parameter, $\mathcal{C}$, of the error term varying over a range 0.0001 - 100. While training our CNN, different values of alpha (in ReLU) and activation functions were used to find the most accurate model. In addition, increase in the number of hidden layers resulted in very little change in accuracy. As the result, the Convolutional Neural Network (CNN) (with L2 reg. term = 0.001 and 4 hidden layers) was found to produce the best results with a classification accuracy of 94.73%, a slightly poorer accuracy of 75.83% with a radial basis function (RBF) (with $\mathcal{C}$ = 100), and the least accurate was the Logistic Regression with accuracy of 41.85%. Index Terms—Logistic, Neural Networks, Convolution, SVM.

This code accompanies the research paper:

### **[Arabic Handwritten Characters Classification Using Logistic Regression, SVM, and Neural Networks](Arabic_Handwritten_Characters_ML_Final_Project.pdf)**

Machine Learning Final Project  
Henrikas Krukauskas, Almadi Shiryayev  
New York University, 2022.
