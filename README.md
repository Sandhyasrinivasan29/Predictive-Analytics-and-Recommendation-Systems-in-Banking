# Predictive-Analytics-and-Recommendation-Systems-in-Banking

# Problem Statement:

Banks handle a lot of customer data and transactions every day. To improve customer service, reduce risks, and offer better products, this project focuses on three main goals:

1. Loan Default Prediction: Using data to identify loans at risk of not being repaid.
2. Customer Segmentation: Grouping customers based on their transaction habits.
3. Product Recommendations: Suggesting the right banking products to customers.
# Technologies Used:
* Programming Language: Python

 Libraries and Frameworks:
  * Data Manipulation: Pandas, NumPy
  * Machine Learning: Scikit-learn, Surprise
  * Visualization: Matplotlib, Seaborn
  * Web Application: Streamlit
  * Serialization: Pickle

# Installation Commands

  * pip install pandas
  * pip install numpy
  * pip install scikit-learn
  * pip install matplotlib
  * pip install seaborn
  * pip install streamlit
  * pip install scikit-surprise
# Data Preparation:
Synthetic data is generated using the Faker Python library.
1. Customer Information: Customer_id, Age, Gender, Income, Credit_score, Credit_score_category, Debit_income
2. Loan Information: Loan_amount, Interest_rate, Loan_term, Loan_type, Repayment_status
3. Transaction Details: Transaction_id, Transaction_amt, Transaction_type, Transaction_data, Transaction_month, Transaction_year, Transaction_freq
4. Interaction Details: Product_id, Product_names, Interaction_type, Interaction_date, Interaction_month, Interaction_year

# Exploratory Data Analysis
1. Visualize Outliers and Skewness with Boxplot,Histplot.
2. Analyze and Treat Skewness using log, sqrt, box-cox.

# Model Building and Evaluation Process
1. Data Splitting:
* Train-Test Split: Divide the data into training and testing sets to ensure proper evaluation of models. Typically, a 70-30 or 80-20 split is used for training and testing.
* Cross-Validation: Use k-fold cross-validation to ensure that the model performs consistently across different subsets of the data.
2. Model Training and Evaluation:
Loan Default Prediction:
Objective: Predict whether a customer will default on their loan (classification task).
Algorithms:
* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
# Evaluation Metrics:
* Accuracy: Measures the overall correctness of the model.
* Precision: Measures the proportion of true positive predictions out of all positive predictions.
* Recall: Measures the proportion of true positive predictions out of all actual positive cases.
* F1 Score: The harmonic mean of Precision and Recall, useful for imbalanced datasets.
# Customer Segmentation:
Objective: Group customers based on their transaction behavior to identify patterns and tailor services.
# Algorithms:
* KMeans Clustering
* Hierarchical Clustering
# Evaluation Metrics:
* Silhouette Score: Measures how similar each point is to its own cluster compared to other clusters.
* Davies-Bouldin Index: Evaluates the compactness and separation between clusters; lower scores are better.
# Product Recommendations:
Objective: Recommend products to users based on their interaction history.
# Algorithms:
* Collaborative Filtering (User-based or Item-based)
* Content-Based Filtering
# Evaluation Metrics:
* Precision: Measures the proportion of recommended products that are relevant.
* Recall: Measures the proportion of relevant products that were recommended.
* Mean Average Precision (MAP): Evaluates the ranking of recommended items based on relevance.
* Normalized Discounted Cumulative Gain (NDCG): Evaluates the ranking of recommendations with higher relevance being more important at the top.
3. Hyperparameter Tuning and Optimization:
* Cross-Validation: Utilize cross-validation techniques to improve the generalization of models by evaluating them on multiple subsets of the data.
* Grid Search: Perform grid search for hyperparameter tuning to find the best parameters for the model and enhance its performance.
# Streamlit website
* Develop a Streamlit App for interactive predictions, customer segmentations and product recommendations.
* Allow the users to input feature values and display predictions, customer segmentations and product recommendations.

