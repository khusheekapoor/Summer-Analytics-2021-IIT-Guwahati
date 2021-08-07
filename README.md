# Summer-Analytics-2021-IIT-Guwahati

Summer Analytics 2021 was a Data Science Bootcamp organized by CCA, IIT Guwahati.

The course spanned over 6 weeks and was power packed with the concepts and fundamentals of Data Science, Machine Learning, and Deep Learning. The Capstone Hackathon at the end enabled me to test the acquired skills hands-on. All the effort was worth it!

An outline of the week-wise graded and ungraded assignments:

## Week-1: Introduction to Python & Python Libraries

The topics under Week-1 included Basic Python Programming, Data Analysis with NumPy and Pandas, Visualizations with Matplotlib and Seaborn, Data Types in Statistics, Measures of Central Tendency, Range, Standard Deviation, Box Plots and Outliers.

### Ungraded Assignments:
[1:](/Ungraded&#32;Assignments/Basic_Data_Analysis_Assignment_1.ipynb) Performing and implementing basic Exploratory Data Analysis functions and techniques on a dataset using Pandas.
 
[2:](/Ungraded&#32;Assignments/Basic_Data_Analysis_Assignment_2.ipynb) Performing and implementing basic Exploratory Data Analysis functions and techniques on a dataset using NumPy and Pandas.
 
 ### [Graded Assignment:](/Graded&#32;Assignments/Week_1.ipynb)
Performing and implementing basic Exploratory Data Analysis functions and techniques on a dataset using NumPy, Pandas and Seaborn.
<br></br>

## Week-2: Machine Learning Algorithms
Week-2 covered the concepts of Outlier Analysis, Handling Missing Values, introduction to Supervised, Unsupervised and Reinforcement Learning, Linear Regression with One and Multiple Variables with and without Scikit-Learn, and Logistic Regression with and without Scikit-Learn.

### [Ungraded Assignment:](/Ungraded&#32;Assignments/Basic_Machine_Learning_Assignment.ipynb)
Creating basic Linear and Logistic Regression models and finding the Mean Squared Error and Accuracy score after applying Square Root Transformation on a skewed dataset.

### [Graded Assignment:](/Graded&#32;Assignments/Week_2.ipynb)
Implementing the Linear Regression algorithm for prediction and computing the Cost Function which is optimised by Gradient Descent.
<br></br>
## Week-3: Model Tuning
The contents of Week-3 spanned over Bias-Variance Tradeoff, L1/Lasso and L2/Ridge Regularization, Support Vector Machine, Feature Transformation & Scaling, Label Encoding and One-Hot Encoding, Evaluation Metrics viz. R2 Score, Adjusted R2 Score, F1 Score, Precision, Recall, Accuracy, etc., and Visual Evaluation Metrics viz. Confusion Matrix and AUC-ROC Curve.

### [Ungraded Assignment:](/Ungraded&#32;Assignments/Model_Tuning.ipynb)
Tuning a basic Linear Regression model by Filling Null and NaN values, Scaling, Encoding, and Transforming Features, implementing L1 and L2 Regularization (Lasso and Ridge Regression) and Cross Validation, and finding and calculating R2 score, Mean Absolute Error and Mean Squared Error.

### [Graded Assignment:](/Graded&#32;Assignments/Week_3.ipynb)
Predicting the class of the dependent variable using different algorithms, tuning the hyperparameters to obtain better results, and evaluating through various evaluation metrics.
<br></br>

## Week-4: Tree-Based Algorithms
In Week-4, we learnt about Decision Trees, Random Forests, Gradient Boosting, and the XGBoost, AdaBoost, CatBoost and LightGBM algorithms.

### Ungraded Assignments:
For Ungraded Assignments in Week-4, we had 2 famous Kaggle Micro-Courses:

[Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)

<img src="https://user-images.githubusercontent.com/74901388/128607679-d69a917f-b25f-4b99-978a-bfca17bff5f8.png" width="500" height="300" />

[Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning)

<img src="https://user-images.githubusercontent.com/74901388/128607806-314474f2-c892-4b66-b9a0-85dbf7831c88.png" width="500" height="300" />

### [Graded Assignment:](Graded&#32;Assignments/Week_4.ipynb)
Predicting the class of the dependent variable using tree-based and ensemble algorithms like Decision Trees, Random Forests, Xtreme Gradient Boost, AdaBoost and other classification algorithms like Gaussian Naive Bayes, Logistic Regression and K-Nearest Neighbors, calculating the Accuracy and plotting the Confusion Matrix, after analysing the data using different visualizations viz. Correlation Heatmap, Countplot and Boxplot and removing the outliers.
<br></br>

## Week-5: Neural Networks
Alongwith Classification and Regression using Neural Networks with and without Keras, Week-5 included Unsupervised Learning methods viz. K-Means Clustering and Dimensionality Reduction with Principal Component Analysis.

### [Graded Assignment:](/Graded&#32;Assignments/Week_5.ipynb)
Classifying between 'cat' and 'non-cat' pictures using Logistic Regression and Neural Networks by defining the model structure, initializing the model's parameters, calculating current loss (forward propagation) and current gradient (backward propagation), and optimizing the model by updating the parameters using gradient descent.
<br></br>

## Week-6: Capstone Hackathon
The skills learnt in the 5 weeks were put to test in the Capstone Hackathon conducted on DPhi.

### Problem Statement:
Let's take a case where an advertiser on the platform (DeltaX) would like to estimate the performance of their campaign in the future.

Imagine it is the first day of March and you are given the past performance data of ads between 1st August to 28th Feb. You are now tasked to predict an ad's future performance (revenue) between March 1st and March 15th. Well, it is now time for you to put on your problem-solving hats and start playing with the [training data](/Hackathon/Train_Data.csv) and [testing data](/Hackathon/Test_Data.csv) provided.

### [My Model:](/Hackathon/Week_6.ipynb)
I created a Pipeline to calculate the value of the continuous dependent variable using XGBRFRegressor with tuned hyperparameters after transforming the numerical columns using Simple Imputer and Robust Scaler and the categorical columns using Simple Imputer and One Hot Encoder and Engineering new Features. My Final Rank and RMSE on the leaderboard:

![image](https://user-images.githubusercontent.com/74901388/128608298-c8617ae9-c93d-45c1-9845-8e6419b31c39.png)

<br></br>

***All the credits for hosting the bootcamp, providing the graded and ungraded assignments and datasets, and conducting the hackathon go to CCA, IIT Guwahati.***


 




