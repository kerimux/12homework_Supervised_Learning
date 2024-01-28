# 12homework_Supervised_Learning

* Explain the purpose of the analysis.
  
The primary objective of our Supervised Machine Learning Analysis is to train a predictive model using labeled data and subsequently assess its performance on a separate test dataset. Through this analysis, we aim to evaluate the effectiveness of our model in making accurate predictions on unseen data, leveraging metrics such as accuracy, precision, and recall. The ultimate goal is to inform decision-making processes based on the model's ability to generalize well to real-world scenarios, providing insights into its reliability and predictive capabilities.

* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
  
Supervised Machine Learning:
lending_data.csv file containes feature like: 

    'loan_size',
    
    'interest_rate',
    
    'borrower_income',
    
    'debt_to_income', 
    
    'num_of_accounts',
    
    'derogatory_marks',
    
    'total_debt'. 

    
And the target: 

    'loan_status'. 
    
The target variable serves as a label, and the goal is to predict this target using the provided features.
The objective of the predictions is to determine whether a loan is categorized as healthy or high risk. This process involves feeding the algorithm with features to enable it to make accurate predictions regarding the target variable.
Utilizing the value_counts() method on the target variable (y) to summarize the total count of loans in our dataset, revealing counts of 75,036 healthy loans and 2,500 high-risk loans.


* Describe the stages of the machine learning process you went through as part of this analysis.

First Step: Data Preparation.
I initiated the machine learning process by separating our dataset into feature and target variables. The target variable(y) was defined as the entirety of the 'loan_status' column, (y=lending_df['loan_status']). For the feature variable(X) I excluded the target variable by removing the 'loan_status' column, (X=lending_df.drop(['loan_status'], axis=1). 

Second Step: Data Splitting
I separated our features and targets into distinct sets, training and testing datasets (default split is 75% training and 25% testing). During this process, the data was randomly divided into training features, testing features, training targets, and testing targets. To ensure consistency in randomness across multiple runs of the code, we deliberately set the parameter (random_state=1) during the data split. This choice allowed us to obtain the same random split each time the code was executed.

Third Step: Model-Fit-Predict
In this phase, I defined the logistic regression model algorithm, fitting the model with training features and corresponding targets. Subsequently, I leveraged the trained model to make predictions on the testing features. Once the Model-Fit-Predict sequence was completed, I conducted a thorough comparison of the accuracy of predictions against our testing targets.

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
  
In the original dataset, I observed a substantial class imbalance, with 96.77% of the loans categorized as healthy loans and only 3.23% classified as high-risk loans. To mitigate the impact of this imbalance during machine learning model training, I utilized the RandomOverSampler method. By doing so, I ensure that the machine learning model does not disproportionately favor the majority class during training, preventing potential biases and allowing for more effective learning from both classes

## Results
Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
  
Logistic regression model predicts on original dateset is very high accuracy - 0.99.

Healthy loan('0'): precision - 1.00 and recall - 0.99

High-risk loan('1'): precision - 0.85 and recall - 0.91

Confusion Matrix summary: 

True Negative: 18663 (96.15%), 

False Positive: 102 (0.53%), 

False Negative: 56 (0.29%), 

True Positive: 563 (2.92%).


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  
Logistic regression model, fit with oversampled data predicts very high accuracy - 0.99.

Healthy loan('0'): precision - 1.00 and recall - 0.99

High-risk loan('1'): precision - 0.84 and recall - 0.99


Confusion Matrix summary: 
True Negative: 18649 (95.56%), 

False Positive: 116 (0.60%),

False Negative: 4 (0.02%), 

True Positive: 615 (3.82%).


## Summary
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

In comparing of machine learning models, the resampled model demonstrated a slight improvement over the original model. While the precision for high-risk loans decreased marginally by 0.01, the recall increased by 0.08. Notably, the performance for healthy loans remained unchanged. 
In summary: in the context of underwriting new loans, the priority often lies in minimizing the creation of high-risk 'False Positive' loans, even if it means missing some of the healthy loans 'True Positives'. Given this, a model with higher precision is preferable, "True Positive" when down by 0.9% and "False Positive" aslo when down by 0.07%. Accordingly, the Logistic Regression model trained on the original dataset performed better for our specific purposes. 
