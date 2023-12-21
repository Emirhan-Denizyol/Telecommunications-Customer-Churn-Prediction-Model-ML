import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings
warnings.simplefilter(action='ignore', category=Warning)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: '%.3f' % x)

###############################
# LOAD DATASET
###############################

df = pd.read_csv("C:/Users/Emirhan Denizyol/PycharmProjects/Telco_Customer_Churn/Dataset/Telco-Customer-Churn.csv")
df_y = df['Churn']
df = df.drop(['customerID', 'Churn'], axis=1)

###############################
# DATA REVIEW
###############################

print(df.head())
#    gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling              PaymentMethod  MonthlyCharges TotalCharges
# 0  Female              0     Yes         No       1           No  No phone service             DSL             No          Yes               No          No          No              No  Month-to-month              Yes           Electronic check          29.850        29.85
# 1    Male              0      No         No      34          Yes                No             DSL            Yes           No              Yes          No          No              No        One year               No               Mailed check          56.950       1889.5
# 2    Male              0      No         No       2          Yes                No             DSL            Yes          Yes               No          No          No              No  Month-to-month              Yes               Mailed check          53.850       108.15
# 3    Male              0      No         No      45           No  No phone service             DSL            Yes           No              Yes         Yes          No              No        One year               No  Bank transfer (automatic)          42.300      1840.75
# 4  Female              0      No         No       2          Yes                No     Fiber optic             No           No               No          No          No              No  Month-to-month              Yes           Electronic check          70.700       151.65

df.columns
# Index(['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'], dtype='object')

print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 7043 entries, 0 to 7042
# Data columns (total 19 columns):
#  #   Column            Non-Null Count  Dtype
# ---  ------            --------------  -----
#  0   gender            7043 non-null   object
#  1   SeniorCitizen     7043 non-null   int64
#  2   Partner           7043 non-null   object
#  3   Dependents        7043 non-null   object
#  4   tenure            7043 non-null   int64
#  5   PhoneService      7043 non-null   object
#  6   MultipleLines     7043 non-null   object
#  7   InternetService   7043 non-null   object
#  8   OnlineSecurity    7043 non-null   object
#  9   OnlineBackup      7043 non-null   object
#  10  DeviceProtection  7043 non-null   object
#  11  TechSupport       7043 non-null   object
#  12  StreamingTV       7043 non-null   object
#  13  StreamingMovies   7043 non-null   object
#  14  Contract          7043 non-null   object
#  15  PaperlessBilling  7043 non-null   object
#  16  PaymentMethod     7043 non-null   object
#  17  MonthlyCharges    7043 non-null   float64
#  18  TotalCharges      7043 non-null   object
# dtypes: float64(1), int64(2), object(16)
# memory usage: 1.0+ MB

print(df.describe().T)
#                   count   mean    std    min    25%    50%    75%     max
# SeniorCitizen  7043.000  0.162  0.369  0.000  0.000  0.000  0.000   1.000
# tenure         7043.000 32.371 24.559  0.000  9.000 29.000 55.000  72.000
# MonthlyCharges 7043.000 64.762 30.090 18.250 35.500 70.350 89.850 118.750


def species_identification(dataframe, cat_th=5, car_th=10):
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.

    Parameters
    ----------
    dataframe: dataframe
        Variable names are the dataframe to be used
    cat_th: int, float
        Class threshold value for numerical but categorical variables
    car_th: int, float
        Class value for categorical but cardinal variables

    Returns
    -------
    cat_cols: list
        Categorical variable list
    num_cols: list
        Numerical variable list
    cat_but_car: list
        List of cardinal variables with categorical view
    num_but_cat: list
        List of cardinal variables with numerical view

    Notes
    -------
    cat_cols + num_cols + cat_but_car = Total number of variables
    num_but_cat inside cat_cols
    The sum of the 3 lists that return is equal to the total number of variables.
    """

    # cat_cols and cat_but_car

    cat_cols = [col for col in dataframe.columns if
                str(dataframe[col].dtype) in ["category", "bool", "object"]]
    num_but_cat = [col for col in dataframe.columns if
                   str(dataframe[col].dtype) in ["int64", "float64"] and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtype) in ["category", "bool", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations : {dataframe.shape[0]}")
    print(f"Variables : {dataframe.shape[1]}")
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_car : {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = species_identification(df)

# Categorical Variable Analysis


def cat_summary(dataframe, col_name, plot=False):
    """
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenen dataframe'dir.
    col_name: list
        Hesaplanmak istenen değişken listesi
    plot: Boolean
        Grafiğin gösterilip gösterilmemesini belirtme

    Returns
    -------

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Oran": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("########################################################")
    """
    Bu fonksiyonun amacı kolonlarda ki eşsiz değişkenlerin kolonda ki yüzdeliklerini bulmmak
    """

    if plot:
        if df[col_name].dtype == "bool":
            df[col] = df[col_name].astype("int64")
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
        else:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)

#         gender   Oran
# gender
# Male      3555 50.476
# Female    3488 49.524
# ########################################################
#          Partner   Oran
# Partner
# No          3641 51.697
# Yes         3402 48.303
# ########################################################
#             Dependents   Oran
# Dependents
# No                4933 70.041
# Yes               2110 29.959
# ########################################################
#               PhoneService   Oran
# PhoneService
# Yes                   6361 90.317
# No                     682  9.683
# ########################################################
#                   MultipleLines   Oran
# MultipleLines
# No                         3390 48.133
# Yes                        2971 42.184
# No phone service            682  9.683
# ########################################################
#                  InternetService   Oran
# InternetService
# Fiber optic                 3096 43.959
# DSL                         2421 34.375
# No                          1526 21.667
# ########################################################
#                      OnlineSecurity   Oran
# OnlineSecurity
# No                             3498 49.666
# Yes                            2019 28.667
# No internet service            1526 21.667
# ########################################################
#                      OnlineBackup   Oran
# OnlineBackup
# No                           3088 43.845
# Yes                          2429 34.488
# No internet service          1526 21.667
# ########################################################
#                      DeviceProtection   Oran
# DeviceProtection
# No                               3095 43.944
# Yes                              2422 34.389
# No internet service              1526 21.667
# ########################################################
#                      TechSupport   Oran
# TechSupport
# No                          3473 49.311
# Yes                         2044 29.022
# No internet service         1526 21.667
# ########################################################
#                      StreamingTV   Oran
# StreamingTV
# No                          2810 39.898
# Yes                         2707 38.435
# No internet service         1526 21.667
# ########################################################
#                      StreamingMovies   Oran
# StreamingMovies
# No                              2785 39.543
# Yes                             2732 38.790
# No internet service             1526 21.667
# ########################################################
#                 Contract   Oran
# Contract
# Month-to-month      3875 55.019
# Two year            1695 24.066
# One year            1473 20.914
# ########################################################
#                   PaperlessBilling   Oran
# PaperlessBilling
# Yes                           4171 59.222
# No                            2872 40.778
# ########################################################
#                            PaymentMethod   Oran
# PaymentMethod
# Electronic check                    2365 33.579
# Mailed check                        1612 22.888
# Bank transfer (automatic)           1544 21.922
# Credit card (automatic)             1522 21.610
# ########################################################
#                SeniorCitizen   Oran
# SeniorCitizen
# 0                       5901 83.785
# 1                       1142 16.215
# ########################################################


# Numerical Variable Analysis


def num_summary(dataframe, numerical_col, plot=False):
    quantile = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantile).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.ylabel(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col)

# count   7043.000
# mean      32.371
# std       24.559
# min        0.000
# 5%         1.000
# 10%        2.000
# 20%        6.000
# 30%       12.000
# 40%       20.000
# 50%       29.000
# 60%       40.000
# 70%       50.000
# 80%       60.000
# 90%       69.000
# 95%       72.000
# 99%       72.000
# max       72.000
# Name: tenure, dtype: float64
# count   7043.000
# mean      64.762
# std       30.090
# min       18.250
# 5%        19.650
# 10%       20.050
# 20%       25.050
# 30%       45.850
# 40%       58.830
# 50%       70.350
# 60%       79.100
# 70%       85.500
# 80%       94.250
# 90%      102.600
# 95%      107.400
# 99%      114.729
# max      118.750
# Name: MonthlyCharges, dtype: float64


# MISSING VALUES

print(df.isnull().sum())
# gender              0
# SeniorCitizen       0
# Partner             0
# Dependents          0
# tenure              0
# PhoneService        0
# MultipleLines       0
# InternetService     0
# OnlineSecurity      0
# OnlineBackup        0
# DeviceProtection    0
# TechSupport         0
# StreamingTV         0
# StreamingMovies     0
# Contract            0
# PaperlessBilling    0
# PaymentMethod       0
# MonthlyCharges      0
# TotalCharges        0
# dtype: int64

# OUTLIER


def outliers_threshold(dataframe, col_name, q1=0.25, q3=0.75, floor_number=1.5):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    iqr = quantile3 - quantile1
    upper_limit = quantile3 + floor_number * iqr
    lower_limit = quantile1 - floor_number * iqr
    return lower_limit, upper_limit


for col in num_cols:
    print(col, outliers_threshold(df, col))

# tenure (-60.0, 124.0)
# MonthlyCharges (-46.02499999999999, 171.375)

"""
tenure;
    low limit = -105.5
    up limit = 178.5
    Not: Since the month period used will not be negative, it would be good to check here.
    
MonthlyCharges;
    low limit = -111.975 
    up limit = 239.025
    Not: Since the monthly salary will not be negative, it is worth checking here as well.


"""

# There does not appear to be any negativity in either of the two controls.

for col in num_cols:
    for i in df[col]:
        if i < 0:
            print(col, i)


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outliers_threshold(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))

# tenure False
# MonthlyCharges False


def grab_outliers(dataframe, col_name, index=False):
    low_limit, up_limit = outliers_threshold(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))].head())
    else:
        print(dataframe[((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))])

    if index:
        return dataframe[((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit))].index


for col in num_cols:
    grab_outliers(df, col)

# Empty DataFrame
# Columns: [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
# Index: []
# Empty DataFrame
# Columns: [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
# Index: []

##################################################
# ENCODING and STANDARDIZATION
##################################################


def categoric_encoding(dataframe, categorical_columns):
    """
    We encode columns of categorical variables for use in machine language.

    Parameters
    ----------
    dataframe: dataframe
        Variable names are the dataframe to be used
    categorical_columns: list
        Categorical columns you want to encode

    Returns
    -------
    dataframe: dataframe
    Notes
    -------
    """
    for col in categorical_columns:
        if (dataframe[col].nunique() < 3) and (dataframe[col].nunique() > 1):
            dataframe = pd.get_dummies(data=dataframe, columns=[col], drop_first=True, dtype='int64')

        elif dataframe[col].nunique() > 2:
            dataframe = pd.get_dummies(data=dataframe, columns=[col], drop_first=False, dtype='int64')
    return dataframe


df = categoric_encoding(df, cat_cols)

# NORMALIZING CATEGORICAL BUT CARDINAL VALUES

df.info()


def convert_categoric_but_cardinal(dataframe, categoric_but_cardinal):
    """
    We encode columns of variables that are categorical but cardinal for use in machine language.

    Parameters
    ----------
    dataframe: dataframe
        Variable names are the dataframe to be used
    categoric_but_cardinal: list
        Categorical but cardinal columns you want to encode

    Returns
    -------
    dataframe: dataframe
    Notes
    -------
        Function step by step:

        1. Loops the specified categoric_but_cardinal columns.
        2. Replaces the whitespaces in each column with NaN values.
        3. It then tries to convert the column to a numeric data type using pd.to_numeric. In case of error (for example, if there is a space character that cannot be converted to a numeric value), it uses the errors='coerce' parameter to handle this situation.
        4. After converting the column to a numeric data type, it replaces the spaces with actual numeric values for the columns containing the spaces.
    """
    for col in categoric_but_cardinal:
        dataframe[col] = pd.to_numeric(dataframe[col].replace(' ', pd.NaT), errors='coerce').astype('float64')
        dataframe[col] = dataframe[col].replace(' ', '', regex=True)

    return dataframe


df = convert_categoric_but_cardinal(df, cat_but_car)

print(df.isnull().sum())
# tenure                                      0
# MonthlyCharges                              0
# TotalCharges                               11
# gender_Male                                 0
# Partner_Yes                                 0
# Dependents_Yes                              0
# PhoneService_Yes                            0
# MultipleLines_No                            0
# MultipleLines_No phone service              0
# MultipleLines_Yes                           0
# InternetService_DSL                         0
# InternetService_Fiber optic                 0
# InternetService_No                          0
# OnlineSecurity_No                           0
# OnlineSecurity_No internet service          0
# OnlineSecurity_Yes                          0
# OnlineBackup_No                             0
# OnlineBackup_No internet service            0
# OnlineBackup_Yes                            0
# DeviceProtection_No                         0
# DeviceProtection_No internet service        0
# DeviceProtection_Yes                        0
# TechSupport_No                              0
# TechSupport_No internet service             0
# TechSupport_Yes                             0
# StreamingTV_No                              0
# StreamingTV_No internet service             0
# StreamingTV_Yes                             0
# StreamingMovies_No                          0
# StreamingMovies_No internet service         0
# StreamingMovies_Yes                         0
# Contract_Month-to-month                     0
# Contract_One year                           0
# Contract_Two year                           0
# PaperlessBilling_Yes                        0
# PaymentMethod_Bank transfer (automatic)     0
# PaymentMethod_Credit card (automatic)       0
# PaymentMethod_Electronic check              0
# PaymentMethod_Mailed check                  0
# SeniorCitizen_1                             0
# dtype: int64

"""
After converting categorical but cardinal variables to numeric type, null values appeared. But it wasn't there when we checked before.
"""

imputer = SimpleImputer(strategy='mean')

df[['TotalCharges']] = imputer.fit_transform(df[['TotalCharges']])

# NORMALIZING NUMERICAL VALUES

num_cols = num_cols + cat_but_car
"""
After examining the categorical but cardinal values, we converted them to numerical type. Then we added the other numerical columns to num_cols.
"""


def numerical_normalizasyonu(dataframe, numerical_columns):
    """
    Since large numerical values in the model we will create will negatively affect us, we standardized them. We did this with RobustScaler.

    Parameters
    ----------
    dataframe: dataframe
        Variable names are the dataframe to be used
    numerical_columns: list
        Numeric columns we want to standardize

    Returns
    -------
    dataframe: dataframe
    Notes
    -------
        Function step by step:

        1. The RobustScaler object (rs) is created.
        2. A loop is started over each numerical column in the specified numerical_columns list.
        3. The fit_transform method is used to standardize each column with RobustScaler and adds the standardized column instead of the original column. The new column name is created by adding "_robust_scaler" to the original column name.
        4. The original numeric column is removed from the dataframe (using the drop method) as its standardized version has been added.
        5. As a result, all numeric columns are standardized with RobustScaler, returning the updated data frame with new columns added and original columns removed.
    """
    rs = RobustScaler()
    for col in numerical_columns:
        dataframe[f'{str(col)}_robust_scaler'] = rs.fit_transform(dataframe[[col]])
        dataframe.drop([col], axis=1, inplace=True)

    return dataframe


df = numerical_normalizasyonu(df, num_cols)

contains_no = any(('No' in str(col_values)) if isinstance(col_values, (str, bytes)) else False for col_values in df.values.flatten())

if contains_no:
    print("DataFrame içinde 'No' ifadesi bulunmaktadır.")
else:
    print("DataFrame içinde 'No' ifadesi bulunmamaktadır.")


####################################################
# MODELLING
####################################################

# 1. Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(df, df_y, test_size=0.2, random_state=42)

reg_model = LogisticRegression().fit(df, df_y)

cv_results = cross_validate(reg_model, X=df, y=df_y, cv=5, scoring=['accuracy', 'roc_auc'])


print(cv_results["test_accuracy"].mean())
# 0.8039197085295825        0.8451635868481733

print(cv_results["test_roc_auc"].mean())
# 0.8451635868481733


y_pred = reg_model.predict(df)
print(classification_report(df_y, y_pred))

#               precision    recall  f1-score   support
#           No       0.85      0.90      0.87      5174
#          Yes       0.66      0.55      0.60      1869
#     accuracy                           0.81      7043
#    macro avg       0.75      0.72      0.74      7043
# weighted avg       0.80      0.81      0.80      7043

y_prob = reg_model.predict_proba(df)[:, 1]
print(roc_auc_score(df_y, y_prob))
# 0.8480553568352112


####################################################
# 2. KNN
####################################################

knn_model = KNeighborsClassifier().fit(df, df_y)

y_pred = knn_model.predict(df)

y_prob = knn_model.predict_proba(df)[:, 1]

print(classification_report(df_y, y_pred))
#               precision    recall  f1-score   support
#           No       0.87      0.91      0.89      5174
#          Yes       0.72      0.63      0.67      1869
#     accuracy                           0.84      7043
#    macro avg       0.80      0.77      0.78      7043
# weighted avg       0.83      0.84      0.83      7043

print(roc_auc_score(df_y, y_prob))
# 0.8972499138074204

knn_model.get_params()
# {'algorithm': 'auto',
# 'leaf_size': 30,
# 'metric': 'minkowski',
# 'metric_params': None,
# 'n_jobs': None,
# 'n_neighbors': 5,
# 'p': 2,
# 'weights': 'uniform'}
knn_params = {'n_neighbors': range(2, 50)}

knn_gs_best = GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose=1).fit(df, df_y)

print(knn_gs_best.best_params_)
# {'n_neighbors': 46}

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(df, df_y)

cv_results = cross_validate(knn_final,
                            X=df,
                            y=df_y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

print(cv_results['test_accuracy'].mean())
# 0.7951154106716561        0.8303074364322954

print(cv_results['test_f1'].mean())
# 0.6170909049720137

print(cv_results['test_roc_auc'].mean())
# 0.8303074364322954


####################################################
# 3. Decision Tree Classification : CART
####################################################

cart_model = DecisionTreeClassifier(random_state=1).fit(df, df_y)

y_pred = cart_model.predict(df)

y_prob = cart_model.predict_proba(df)[:, 1]

print(classification_report(df_y, y_pred))
#               precision    recall  f1-score   support
#           No       1.00      1.00      1.00      5174
#          Yes       1.00      0.99      1.00      1869
#     accuracy                           1.00      7043
#    macro avg       1.00      1.00      1.00      7043
# weighted avg       1.00      1.00      1.00      7043

print(roc_auc_score(df_y, y_prob))
# 0.9999825236401375

##########################
# Hold Out
##########################

X_train, X_test, y_train, y_test = train_test_split(df, df_y, test_size=0.30, random_state=17)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

##########################
# Train Error
##########################

y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
#               precision    recall  f1-score   support
#           No       1.00      1.00      1.00      3635
#          Yes       1.00      0.99      1.00      1295
#     accuracy                           1.00      4930
#    macro avg       1.00      1.00      1.00      4930
# weighted avg       1.00      1.00      1.00      4930

print(roc_auc_score(y_train, y_prob))
# 0.9999929896491107

##########################
# Test Error
##########################

y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support
#           No       0.82      0.81      0.81      1539
#          Yes       0.50      0.51      0.50       574
#     accuracy                           0.73      2113
#    macro avg       0.66      0.66      0.66      2113
# weighted avg       0.73      0.73      0.73      2113

print(roc_auc_score(y_test, y_prob))
# 0.660534579447716

####################################################
# Success Verification with CV
####################################################

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
"""
StratifiedKFold is used to maintain class balance when performing k-fold cross-validation of the dataset.
That is, this cross-validation method balances the class proportions within each layer, 
thus ensuring a balanced class distribution between the training and testing sets in each layer.
"""
cart_model = DecisionTreeClassifier(random_state=17).fit(df, df_y)

cv_results = cross_validate(cart_model,
                            X=df,
                            y=df_y,
                            cv=stratified_kfold,
                            scoring=["accuracy", "roc_auc"])

print(cv_results['test_accuracy'].mean())
# 0.7298019227046907

print(cv_results['test_roc_auc'].mean())
# 0.6551122056384742

####################################################
# Hyperparameter Optimization with GridSearchCV
####################################################

cart_model.get_params()
# {'ccp_alpha': 0.0,
# 'class_weight': None,
# 'criterion': 'gini',
# 'max_depth': None,
# 'max_features': None,
# 'max_leaf_nodes': None,
# 'min_impurity_decrease': 0.0,
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,
# 'min_weight_fraction_leaf': 0.0,
# 'random_state': 17,
# 'splitter': 'best'}

cart_params = {"max_depth": range(1, 11),
               "min_samples_split": range(2, 20)}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(df, df_y)

print(cart_best_grid.best_params_)
# {'max_depth': 5, 'min_samples_split': 2}

print(cart_best_grid.best_score_)
# 0.793980155977805

####################################################
# FINAL MODEL
####################################################

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(df, df_y)

print(cart_final.get_params())
# {'ccp_alpha': 0.0,
# 'class_weight': None,
# 'criterion': 'gini',
# 'max_depth': 5,
# 'max_features': None,
# 'max_leaf_nodes': None,
# 'min_impurity_decrease': 0.0,
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,
# 'min_weight_fraction_leaf': 0.0,
# 'random_state': 17,
# 'splitter': 'best'}

cart_final = cart_final.set_params(**cart_best_grid.best_params_).fit(df, df_y)

cv_results = cross_validate(cart_final,
                            X=df,
                            y=df_y,
                            cv=5,
                            scoring=["accuracy", "roc_auc"])

print(cv_results['test_accuracy'].mean())
# 0.793980155977805

print(cv_results['test_roc_auc'].mean())
# 0.82855849989695  0.793980155977805

###############################
# Random Forest
###############################

rf_model = RandomForestClassifier(random_state=17)

cv_results = cross_validate(rf_model, df, df_y, cv=5, scoring=['accuracy', 'roc_auc'])

print(cv_results["test_accuracy"].mean())
# 0.7892957811794309

print(cv_results["test_roc_auc"].mean())
# 0.8204319142280696

print(rf_model.get_params())
# {'bootstrap': True,
# 'ccp_alpha': 0.0,
# 'class_weight': None,
# 'criterion': 'gini',
# 'max_depth': None,
# 'max_features': 'sqrt',
# 'max_leaf_nodes': None,
# 'max_samples': None,
# 'min_impurity_decrease': 0.0,
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,
# 'min_weight_fraction_leaf': 0.0,
# 'n_estimators': 100,
# 'n_jobs': None,
# 'oob_score': False,
# 'random_state': 17,
# 'verbose': 0,
# 'warm_start': False}

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(df, df_y)

print(rf_best_grid.best_params_)
# {'max_depth': 8, 'max_features': 5, 'min_samples_split': 5, 'n_estimators': 200}

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(df, df_y)

cv_results = cross_validate(rf_final, df, df_y, cv=5, scoring=['accuracy', 'roc_auc'])

print(cv_results["test_accuracy"].mean())
# 0.80448799116072

print(cv_results["test_roc_auc"].mean())
# 0.8459577050271367


def plot_importance(model, feature, num=len(df), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': feature.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout
    plt.show()

    if save:
        plt.savefig('importances.png')


plot_importance(rf_final, df)


def value_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(model,
                                               X,
                                               y,
                                               param_name=param_name,
                                               param_range=param_range,
                                               scoring=scoring,
                                               cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label="Training Score", color="b")
    plt.plot(param_range, mean_test_score, label="Validation Score", color="g")

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show(block=True)


value_curve_params(rf_final, df, df_y, "max_depth", param_range=range(1, 11), scoring="accuracy", cv=10)


###################
# GBM
###################

gmb_model = GradientBoostingClassifier(random_state=17)

print(gmb_model.get_params())
# {'ccp_alpha': 0.0,
# 'criterion': 'friedman_mse',
# 'init': None,
# 'learning_rate': 0.1,             !!!!!!!!!
# 'loss': 'log_loss',
# 'max_depth': 3,                   !!!!!!!!!
# 'max_features': None,             !!!!!!!!!
# 'max_leaf_nodes': None,
# 'min_impurity_decrease': 0.0,
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,           !!!!!!!!!
# 'min_weight_fraction_leaf': 0.0,
# 'n_estimators': 100,              !!!!!!!!!   (optimizasyon sayısı (boost etmek))
# 'n_iter_no_change': None,
# 'random_state': 17,               !!!!!!!!!
# 'subsample': 1.0,                 !!!!!!!!!!
# 'tol': 0.0001,
# 'validation_fraction': 0.1,
# 'verbose': 0,
# 'warm_start': False}
cv_results = cross_validate(gmb_model, df, df_y, cv=5, scoring=['accuracy', 'roc_auc'])

print(cv_results["test_accuracy"].mean())
# 0.8069016549454805

print(cv_results["test_roc_auc"].mean())
# 0.846418663574467

gbm_params = {'learning_rate': [0.01, 0.1],
              'max_depth': [3, 8, 10],
              'n_estimators': [100, 500, 1000],
              'subsample': [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gmb_model, gbm_params, cv=5, verbose=True, n_jobs=-1).fit(df, df_y)

print(gbm_best_grid.best_params_)
# {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1}

gbm_final_model = gmb_model.set_params(**gbm_best_grid.best_params_, random_state=17).fit(df, df_y)

cv_results = cross_validate(gbm_final_model, df, df_y, cv=5, scoring=['accuracy', 'roc_auc'])

print(cv_results["test_accuracy"].mean())
# 0.8069016549454805

print(cv_results["test_roc_auc"].mean())
# 0.846418663574467


"""
                        Accuracy                Roc_Auc
GBM                     0.8069016549454805      0.846418663574467
Random Forest           0.80448799116072        0.8459577050271367
Logistic Regression     0.8039197085295825      0.8451635868481733
CART                    0.793980155977805       0.82855849989695
KNN                     0.7951154106716561      0.8303074364322954

"""
