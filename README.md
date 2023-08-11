# TelcoChurnPrediction

![94357telecom churn](https://github.com/ahmetdzdrr/TelcoChurnPrediction/assets/117534684/e4c2567a-a8f6-45ff-a684-b666434ca458)

Introduction to the Telco Customer Churn Dataset
The provided dataset contains information related to customers of a telecommunications company. It encompasses various attributes that offer insights into the characteristics and behaviors of these customers. The dataset is a valuable resource for conducting analyses and building predictive models to understand and anticipate customer churn.

Dataset Description

The dataset comprises the following columns:

customerID: A unique identifier for each customer.

gender: The gender of the customer.

SeniorCitizen: Indicates if the customer is a senior citizen (1) or not (0).

Partner: Indicates if the customer has a partner (Yes) or not (No).

Dependents: Indicates if the customer has dependents (Yes) or not (No).

tenure: The duration, in months, that the customer has been with the company.

PhoneService: Indicates if the customer has phone service (Yes) or not (No).

MultipleLines: Indicates if the customer has multiple lines (Yes, No, or No phone service).

InternetService: The type of internet service the customer has (DSL, Fiber optic, or No).

OnlineSecurity: Indicates if the customer has online security (Yes, No, or No internet service).

OnlineBackup: Indicates if the customer has online backup (Yes, No, or No internet service).

DeviceProtection: Indicates if the customer has device protection (Yes, No, or No internet service).

TechSupport: Indicates if the customer has tech support (Yes, No, or No internet service).

StreamingTV: Indicates if the customer has streaming TV (Yes, No, or No internet service).

StreamingMovies: Indicates if the customer has streaming movies (Yes, No, or No internet service).

Contract: The contract term of the customer (Month-to-month, One year, Two year).

PaperlessBilling: Indicates if the customer has paperless billing (Yes) or not (No).

PaymentMethod: The method by which the customer makes payments.

MonthlyCharges: The monthly amount charged to the customer.

TotalCharges: The total amount charged to the customer.

Churn: Indicates if the customer churned (Yes) or not (No).


Objective

The main objective of working with this dataset is to understand the factors that influence customer churn and to build predictive models that can help identify potential churners. By analyzing the provided attributes, we can uncover patterns and insights that may guide business decisions aimed at reducing customer churn and improving customer retention.

The dataset's attributes cover a range of customer-related aspects, from demographic information to service subscriptions and payment behavior. Analyzing this dataset can lead to valuable insights into customer behavior and preferences, which can ultimately contribute to more effective customer relationship management and strategic decision-making.

In the following sections, we will explore and preprocess the dataset, perform exploratory data analysis (EDA), and potentially build predictive models to achieve the stated objectives.

![Screenshot_21](https://github.com/ahmetdzdrr/TelcoChurnPrediction/assets/117534684/429a05e4-c184-4b43-90c1-99a18b4b0858)

Configuration Class (CFG) for Data Preprocessing
The CFG class is a configuration class designed to control various data preprocessing steps in a machine learning pipeline. By modifying the attributes within this class, you can easily customize the data preprocessing techniques applied to your dataset before training a machine learning model. Let's explore the attributes and their meanings in more detail.

Outlier Handling

outlier_clipper: If set to True, outliers in the data will be clipped to a specified range.
outlier_remover: If set to True, outliers will be removed from the dataset.
outlier_replacer: If set to True, outliers will be replaced with a central value (e.g., mean or median).
Encoding Techniques

one_hot_encoder: If set to True, categorical variables will be one-hot encoded.
label_encoder: If set to True, categorical variables will be label encoded.
ordinal_encoder: If set to True, ordinal categorical variables will be encoded.
Feature Scaling

min_max_scaler: If set to True, data will be scaled using Min-Max scaling.
robust_scaler: If set to True, data will be scaled using Robust scaling.
standard_scaler: If set to True, data will be scaled using Standard scaling.
How to Use the CFG Class

To utilize the CFG class for data preprocessing, follow these steps:

Modify the attributes within the CFG class according to your preprocessing requirements. For example:

CFG.outlier_clipper = True

CFG.label_encoder = True

CFG.robust_scaler = True

Use the CFG attributes in your data preprocessing pipeline. For instance:

if CFG.outlier_clipper:

Apply outlier clipping codes

if CFG.label_encoder:

Apply label encoding codes

if CFG.robust_scaler:

Apply Robust scaling codes

![download](https://github.com/ahmetdzdrr/TelcoChurnPrediction/assets/117534684/e860a919-63ba-4512-ac68-f56ee7f86914)

Visualizing Null and Non-Null Values
In this section, a visualization is created to illustrate the distribution of null and non-null values for each column in the DataFrame (df). The code provided generates a horizontal bar chart using matplotlib to display the percentage of null and non-null values for each column.

1. Creating Null Value DataFrame

The code begins by calculating the count of null values for each column using the .isnull().sum() method on the DataFrame. The result is transformed into a DataFrame named df_null_values, with columns 'Count', 'Porcentaje_nulos', and 'Porcentaje_no_nulos'. The percentage of null and non-null values is calculated based on the total number of rows.

2. Generating the Bar Chart

The horizontal bar chart is created using the matplotlib library. The plt.subplots() function initializes a figure and axes for plotting. The heights of the bars represent the percentages of null and non-null values for each column.

Two sets of horizontal bars are plotted:

rects1: Represents the percentage of null values (colored in red).
rects2: Represents the percentage of non-null values (colored in orange).
Axis labels, ticks, title, and legend are added to enhance the visualization. The autolabel() function is defined to add percentage labels to the bars.

Finally, the visualization is displayed using plt.tight_layout() and plt.show().


Box Plot Visualization Function and Usage Example
The provided Python function, plot_boxplots, is designed to create box plot visualizations for a set of numeric columns in a DataFrame. Box plots are useful for visualizing the distribution, central tendency, and spread of data within each column.

How It Works

The function calculates the number of rows (nrows) required to accommodate all the specified numeric columns in the plot grid.
It creates a subplot grid with the specified number of rows and ncols.
For each numeric column, a box plot is generated using sns.boxplot, and the resulting plot is placed in the corresponding subplot.
The title of each subplot is set to indicate the column name.

![__results___22_0](https://github.com/ahmetdzdrr/TelcoChurnPrediction/assets/117534684/18e57f68-eccc-416a-8ef5-32e10ba0ca3c)

CORRELATION MATRIX

![__results___23_0](https://github.com/ahmetdzdrr/TelcoChurnPrediction/assets/117534684/cbe92130-915f-4414-bd65-06647bba2e25)


TARGET VISUALIZATION

![__results___24_0](https://github.com/ahmetdzdrr/TelcoChurnPrediction/assets/117534684/88d484b3-420c-436b-982b-f786f5ce9eb7)


FEATURE EXTRACTION VISUALIZATION

![__results___31_0](https://github.com/ahmetdzdrr/TelcoChurnPrediction/assets/117534684/61ea9466-59cc-450c-a8b4-a4228e04ae92)


Data Preprocessing: Outlier Handling

In this section, we define a DataProcessorOutlier class responsible for preprocessing the data to handle outliers based on a given configuration (cfg). This class encapsulates methods to clip, remove, or replace outliers in the input DataFrame.

1. Outlier Clipping (outlier_clipper)

The outlier_clipper method clips the extreme values of numerical columns within the interquartile range (IQR). For each numerical column, it calculates the first and third quartiles (q1_val and q3_val) and then clips the column's values to be within this range.

2. Outlier Removal (outlier_remover)

The outlier_remover method removes outliers by filtering out rows where numerical column values fall outside a predefined range ([q1_val, q3_val]) for each column.

3. Outlier Replacement (outlier_replacer)

The outlier_replacer method replaces outliers with the median value for each numerical column. It utilizes the same IQR-based range as in the previous methods.

After defining these outlier handling methods, the process_data function takes a DataFrame as input and applies the specified outlier handling techniques based on the provided configuration (cfg).

Finally, an instance of the DataProcessorOutlier class is created with a configuration object (CFG()), and the process_data method is called on the input DataFrame df.




Data Preprocessing: Encoding Categorical Variables

In this section, we define a DataProcessorEncode class responsible for encoding categorical variables within a given DataFrame. The class offers different encoding techniques based on the provided configuration (cfg).

1. One-Hot Encoding (one_hot_encoder)

The one_hot_encoder method applies one-hot encoding to categorical variables with more than two unique values. It identifies such columns (object_cols) and creates binary columns for each unique category using the pd.get_dummies function. If applicable columns are found, one-hot encoding is applied; otherwise, a warning is issued.

2. Label Encoding (label_encoder)

The label_encoder method applies label encoding to categorical variables with exactly two unique values. It uses the LabelEncoder from scikit-learn to transform binary categorical columns into numerical values (0 and 1). Similar to one-hot encoding, a warning is issued if no applicable columns are found.

3. Ordinal Encoding (ordinal_encoder)

The ordinal_encoder method performs ordinal encoding for all categorical columns. It maps unique category values to incremental integer values, effectively transforming categorical variables into ordinal representations. A dictionary (ordinal_encoder) is created to store the mapping for each categorical column.

After defining these encoding methods, the encode_data function takes a DataFrame as input and applies the specified encoding techniques based on the provided configuration (cfg).

Finally, an instance of the DataProcessorEncode class is created with a configuration object (CFG()), and the encode_data method is called on the input DataFrame df. The encoded DataFrame df is returned and displayed using the .head() method.




Data Preprocessing: Feature Scaling

In this section, we define a DataProcessorScaled class responsible for scaling numerical features within a given DataFrame. The class provides different scaling techniques based on the provided configuration (cfg).

1. Min-Max Scaling (min_max_scaler)

The min_max_scaler method applies Min-Max scaling to numerical columns. It identifies numerical columns (num_cols) and scales their values to the range [0, 1] using the MinMaxScaler from scikit-learn. If applicable numerical columns are found, Min-Max scaling is applied; otherwise, a warning is issued.

2. Standard Scaling (standard_scaler)

The standard_scaler method applies Standard Scaling (z-score normalization) to numerical columns. Similar to Min-Max scaling, it identifies numerical columns and scales their values to have zero mean and unit variance using the StandardScaler from scikit-learn.

3. Robust Scaling (robust_scaler)

The robust_scaler method applies Robust Scaling to numerical columns. This technique is less sensitive to outliers than Min-Max or Standard Scaling. It scales the data using the median and interquartile range. Numerical columns are identified, and their values are scaled using the RobustScaler from scikit-learn.

After defining these scaling methods, the scaled_data function takes a DataFrame as input and applies the specified scaling techniques based on the provided configuration (cfg).

Finally, an instance of the DataProcessorScaled class is created with a configuration object (CFG()), and the scaled_data method is called on the input DataFrame df. The scaled DataFrame df is returned and displayed using the .head() method.


Handling Class Imbalance with SMOTE
In this section, we address the issue of class imbalance in the dataset by employing the Synthetic Minority Over-sampling Technique (SMOTE). SMOTE is a technique used to balance the distribution of the classes by generating synthetic samples for the minority class.

First, we import the necessary library, which is SMOTE from an undisclosed package (presumably imblearn.over_sampling). We initialize the SMOTE instance, setting the random_state to 42 to ensure reproducibility.

Next, we apply SMOTE to the training data. The fit_resample method of the SMOTE instance is used for this purpose. It takes the original X_train (independent variables) and y_train (target variable) as inputs and returns resampled versions, X_train_resampled and y_train_resampled, respectively.

![Screenshot_22](https://github.com/ahmetdzdrr/TelcoChurnPrediction/assets/117534684/e73118be-8fee-447b-9b22-297943da39de)


Model Training, Evaluation, and Result Visualization
In this section, a function named model is defined to train, evaluate, and visualize the performance of a set of machine learning models. Additionally, another function named model_to_dataframe is provided to summarize the results of the model evaluations in a structured DataFrame.

1. plot_confusion_matrix Function

This function, plot_confusion_matrix, takes true labels (y_true), predicted labels (y_pred), the model's name (model_name), and an axis (ax) to create and display a confusion matrix heatmap. The confusion matrix visually represents the distribution of correct and incorrect predictions for each class.

2. model Function

The model function performs the following steps for each model specified in the models dictionary:

Fits the model using the resampled training data (X_train_resampled and y_train_resampled).
Generates predictions (y_pred) on the test set.
Calculates accuracy, precision, recall, and F1 score using various scoring metrics.
Stores the evaluation metrics in the results dictionary.
Calls the plot_confusion_matrix function to visualize the confusion matrix for the current model.
Finally, it displays a grid of confusion matrix heatmaps for each model using matplotlib and returns the results dictionary containing the evaluation metrics.

3. model_to_dataframe Function

This function, model_to_dataframe, utilizes the model function to evaluate the models' performance and converts the resulting results dictionary into a structured DataFrame. The DataFrame includes columns for 'Accuracy', 'Precision', 'Recall', and 'F1 Score' for each model.

The models are trained, evaluated, and ranked based on their accuracy in descending order. The resulting DataFrame provides an overview of the performance of different machine learning models on the test set

![__results___43_0](https://github.com/ahmetdzdrr/TelcoChurnPrediction/assets/117534684/34468168-bd98-4a88-ac9a-0ce828fb686d)

![Screenshot_23](https://github.com/ahmetdzdrr/TelcoChurnPrediction/assets/117534684/307c88aa-e66a-4343-ad2f-d11fd1ffb79b)


Hyperparameter Tuning Function
This function, named hyperparameter_tuning, is designed for performing hyperparameter tuning for a collection of machine learning models. It takes a dictionary of models along with corresponding params (hyperparameter grids), input features X, target variable y, and an optional cv (cross-validation) parameter.

The function iterates through each model specified in the models dictionary and performs hyperparameter tuning if hyperparameters are provided in the params dictionary. For each model, it follows these steps:

If the model doesn't require hyperparameters, it skips tuning for that model and proceeds to the next one.

If the model is an instance of CatBoostClassifier, it sets the verbosity level to 0 to suppress intermediate output during training.

It uses GridSearchCV to perform cross-validated grid search over the specified hyperparameter grid (params[model_name]) for the given model. The cv parameter controls the number of folds in cross-validation, and n_jobs=-1 indicates that computations are parallelized across all available CPU cores.

The best parameters, best score, and best model obtained from the grid search are stored in the best_models dictionary.

The function prints the best parameters and best score for the current model.

Finally, the function returns a dictionary (best_models) containing the best parameters and scores for the models that were subject to hyperparameter tuning.

Note: Ensure that you have the required libraries imported (GridSearchCV, CatBoostClassifier, etc.) and the necessary configurations set before calling this function.

![Screenshot_24](https://github.com/ahmetdzdrr/TelcoChurnPrediction/assets/117534684/802bbca2-ae2c-4914-91a1-b2742b4caf74)


FEATURE IMPORTANCE

![__results___48_0](https://github.com/ahmetdzdrr/TelcoChurnPrediction/assets/117534684/45555330-4a0c-4575-a817-3c49802a9a52)














