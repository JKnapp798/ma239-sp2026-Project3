MA239 Project 3 – Wine Quality Analysis

Author: Jake Knapp

Project Overview

This project analyzes a wine quality dataset using Principal Component Analysis (PCA) and predictive modeling techniques. The goal is to explore the dataset, reduce dimensionality, and predict wine quality based on physicochemical properties.

The analysis covers:

Data cleaning and missing value handling, feature scaling, dimensionality reduction with PCA, model training, 
evaluation with multiple regression techniques, model comparison, and condition number analysis.

Usage:  
pip install pandas numpy matplotlib scikit-learn  
python WineQualityAnalysis.py

Wine Quality Report Requirement Summary:  

Problem Statement   
Q: What are you predicting?  
A: Predicting the quality of wine based on various chemical properties   

Q: Why is this problem meaningful?  
A: Understanding wine quality helps producers improve products and educate customers.  

Dataset Description  
Q: Number of observations  
A: 1143 Wine samples   
Q: Number of features   
A: 12 features measured for prediction, excluding Id and quality  
Q: Types of variables  
A: All features are numeric (target is integer-based)  

Data Cleaning Decisions  
Q: How missing values were handled  
A: Missing values were filled with the column mean. Also note that the original dataset did not feature missing data, so for the purpose of the assignment, random data was erased manually.   
Q: Why you chose that strategy  
A: The dataset is completely numeric; only the mean was needed  
Q: How categorical variables were encoded  
A: This set features no categorical variables, so no one-hot encoding was used.  
Q: Why scaling was necessary   
A: Due to features having vastly different units and scales (such as density and pH), scaling is required. This ensures that the PCA and other models would not lean from the higher-magnitude features.   
PCA Analysis  
Q: Singular value interpretation  
A: Each single value corresponds to the variance demonstrated by a principal component. Larger singular values indicate the direction of high variability in the data.
Q: Explain the variance plot  
A: A cumulative explained variance plot shows that 9 principal components capture 95% of all variance.  
Q: Justification for chosen k  
A: k = 9 was chosen because it captures 95% of the variance. This reduced dimensionality while retaining most information.  
Q: Discussion of dimensionality reduction.   
A: PCA reduced redundant information. Additionally, allows models to focus on the most important components in computations.   

Linear Algebra Analysis  
Q: The normal equation  
A: This is called during the “.fit” command. The normal equation is used to solve Linear Regression through computation.   
Q: Why pseudo-inverse was used  
A: Pseudo-inverse handled potential non-invertible data in X^T * X  
Q: The condition number  
A: In both Linear and Ridge regression, the condition number was about 7.3, demonstrating well-conditioned data with stable coefficients.  
Q: Whether multicollinearity appears to be present  
A: In this demonstration, no major multicollinearity was observed. As the condition number is low, features are not highly correlated after being scaled.  

Results And Interpretation  
Q: Which model performed best?  
A: The Random Forest Regressor performed best, achieving the lowest RMSE (0.546), lowest MAE (0.412), and highest R² (0.464).  
Q: Whether PCA improved performance  
A: PCA was mostly just for analysis and complexity reduction. In this case, the PCA did not have much improvement in predictive results. Overall, the PCA did reduce feature redundancy, making the models more stable.  
Q: Whether the condition number was large   
A: The condition number was relatively small (~7.3).  
Q: What does that imply about feature dependence?  
A: A low condition number means little multicollinearity, meaning the features are not highly correlated after scaling. Linear regression coefficients are reliable.   
Q: What model you would recommend in a real setting?  
A: In practice, I would use the Random Forest Regressor. This is due to it having the best predictive accuracy.   

