# COMPARATIVE-ANALYIS-OF-CLASSIFICTION-ALGORITHM-FOR-LUNG-CANCER-PREDICTION
This project compares Logistic Regression, KNN, Decision Tree, SVM, and Naive Bayes for lung cancer prediction. Decision Tree and SVM achieved the highest accuracy (97.06%), followed by Logistic Regression (96.08%). Models were evaluated using accuracy, precision, recall, and F1 score. 🚀

### **Lung Cancer Prediction Using Machine Learning: Complete Explanation**  

#### **1. Introduction**  
Lung cancer is one of the deadliest diseases worldwide, making early detection crucial for improving survival rates. In this project, we apply **machine learning classification models** to predict lung cancer using a dataset obtained from **Data World**. The dataset consists of patient records with attributes such as **age, gender, smoking history, coughing, chest pain, shortness of breath, and other symptoms**, with the target variable indicating **whether a patient has lung cancer (Yes/No)**.  

The goal of this study is to compare the performance of **five classification algorithms** and determine the best-performing model based on multiple evaluation metrics. The models used in this study are:  
- **Logistic Regression**  
- **K-Nearest Neighbors (KNN)**  
- **Decision Tree**  
- **Support Vector Machine (SVM)**  
- **Naive Bayes**  

#### **2. Dataset Overview**  
The dataset consists of **several patient attributes** that serve as potential indicators for lung cancer. The features include:  
- **Demographic Features**: Age, Gender  
- **Lifestyle Factors**: Smoking history  
- **Symptoms**: Coughing, Wheezing, Chest Pain, Shortness of Breath, Fatigue  
- **Target Variable**: Presence of Lung Cancer (Yes/No)  

Before model training, **data preprocessing** was performed, including **handling missing values, encoding categorical variables, and feature scaling** where necessary. The dataset was then split into **training (80%) and testing (20%) sets** for model evaluation.  

#### **3. Machine Learning Models Used**  
The following five classification models were implemented and tested on the dataset:  

##### **1. Logistic Regression**  
A statistical model used for binary classification. It works by estimating the probability that an instance belongs to a particular class using a sigmoid function.  

##### **2. K-Nearest Neighbors (KNN)**  
A distance-based classifier that predicts the class of a new data point based on the majority class of its nearest neighbors. It is simple but can be sensitive to noisy data.  

##### **3. Decision Tree**  
A tree-based model that splits data at each node based on feature importance, making it easy to interpret. Decision Trees are powerful but can overfit if not properly tuned.  

##### **4. Support Vector Machine (SVM)**  
A model that finds the optimal hyperplane to separate different classes. It is effective for high-dimensional datasets and works well with a clear margin of separation.  

##### **5. Naive Bayes**  
A probabilistic classifier based on Bayes’ theorem, assuming independence between features. It is computationally efficient and works well with categorical data.  

#### **4. Model Performance Evaluation**  
Each model was evaluated using the following performance metrics:  
- **Accuracy**: Measures overall correctness of predictions.  
- **Precision**: Measures how many predicted positives were actually positive.  
- **Recall (Sensitivity)**: Measures how well the model identifies actual positive cases.  
- **F1 Score**: Harmonic mean of Precision and Recall, useful for imbalanced datasets.  

The results of the model evaluation are summarized in the table below:  

| Model                  | Accuracy  | Precision | Recall  | F1 Score |
|------------------------|----------|-----------|---------|---------|
| **Logistic Regression** | 96.08%   | 96.91%    | 98.95%  | 97.92%  |
| **K-Nearest Neighbors** | 93.13%   | 94.89%    | 97.89%  | 96.37%  |
| **Decision Tree**       | 97.06%   | 97.92%    | 98.95%  | 98.43%  |
| **Support Vector Machine (SVM)** | 97.06% | 98.94% | 97.89% | 98.41% |
| **Naive Bayes**        | 95.10%   | 96.87%    | 97.89%  | 97.38%  |

#### **5. Results and Analysis**  
- **Decision Tree and SVM achieved the highest accuracy of 97.06%**, making them the best-performing models in this study.  
- **Logistic Regression performed well with 96.08% accuracy**, making it a strong candidate due to its simplicity and interpretability.  
- **Naive Bayes and KNN showed slightly lower accuracy (95.10% and 93.13%, respectively)** but still provided reliable predictions.  
- The **Confusion Matrix** for each model was analyzed to understand the distribution of false positives and false negatives.  

![image](https://github.com/user-attachments/assets/65d84403-db1a-4a0a-8281-b6184497b94f)

#### **6. Conclusion**  
This study demonstrates the effectiveness of machine learning algorithms in **predicting lung cancer based on patient attributes**. The results show that **Decision Tree and SVM models performed the best with an accuracy of 97.06%**, followed closely by **Logistic Regression (96.08%)**. The high precision, recall, and F1 scores indicate that these models can effectively classify lung cancer cases.  

Future work could involve **testing on larger datasets, using deep learning techniques, or optimizing hyperparameters** to improve performance further. This project highlights the importance of machine learning in **early lung cancer detection**, which could potentially aid medical professionals in diagnosis and treatment planning. 🚀

Runtime Difference – Short Note
The runtime difference between Support Vector Machine (SVM) and Decision Tree was analyzed in terms of training time and prediction time.

Training Time: SVM takes longer to train due to complex mathematical computations and optimization processes, while Decision Tree trains faster using a greedy approach.
Prediction Time: SVM is slower in making predictions as it calculates distances from support vectors, whereas Decision Tree is significantly faster, traversing a tree structure in O(log n) time complexity.
Visualization: A bar chart comparison showed that Decision Tree is computationally more efficient than SVM, making it a better choice for real-time applications.
Thus, Decision Tree is preferred over SVM in scenarios where speed and efficiency are critical, while SVM may still be useful for tasks requiring high accuracy despite its slower runtime. 
![image](https://github.com/user-attachments/assets/3327a665-c79d-4d3f-aae6-85d9308e6e17)
