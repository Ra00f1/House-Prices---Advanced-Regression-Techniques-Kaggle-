import sklearn.linear_model as lm
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def load_data():
    data = pd.read_csv('Data/train.csv')
    test = pd.read_csv('Data/test.csv')
    return data, test


def Process_Data(data):
    # Changing the data type of the columns to category and then to numerical values for the model
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].astype('category')
            data[column] = data[column].cat.codes

    # Filling the missing values with the mean of the column
    data = data.fillna(0)

    # Removing the ID column as it is not useful for the model
    data = data.drop('Id', axis=1)

    # Splitting the data into features and labels
    X_train = data.drop('SalePrice', axis=1)
    y_train = data['SalePrice']

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)

    return X_train, y_train


def Process_Test_Data(test):
    # Changing the data type of the columns to category and then to numerical values for the model
    for column in test.columns:
        if test[column].dtype == 'object':
            test[column] = test[column].astype('category')
            test[column] = test[column].cat.codes

    # Filling the missing values with the mean of the column
    test = test.fillna(0)

    # Removing the ID column as it is not useful for the model
    test = test.drop('Id', axis=1)

    print("Test shape: ", test.shape)

    return test


def Visualize_Data(data):
    plt.figure(figsize=(7, 6))
    sns.distplot(data['SalePrice'], color='g', bins=100)
    plt.show()


def Data_Summary(data):
    print(data.describe())
    print(data.info())
    print(data.count())
    print(data.isnull().sum())

    print(data.shape)
    print(data.nunique())
    print(data['SalePrice'].describe())


def Machine_Learning(X_train, y_train):
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Logistic Regression
    log_reg = lm.LogisticRegression()
    log_reg.fit(X_train, y_train)
    log_reg_score = log_reg.score(X_test, y_test)
    print("Logistic Regression Testing Accuracy: ", log_reg_score)

    lin_reg = lm.LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_reg_score = lin_reg.score(X_test, y_test)
    print("Linear Regression Testing Accuracy: ", lin_reg_score)

    Sgd_reg = lm.SGDRegressor()
    Sgd_reg.fit(X_train, y_train)
    Sgd_reg_score = Sgd_reg.score(X_test, y_test)
    print("SGD Testing Accuracy: ", Sgd_reg_score)

    Ridge_reg = lm.Ridge()
    Ridge_reg.fit(X_train, y_train)
    Ridge_reg_score = Ridge_reg.score(X_test, y_test)
    print("Ridge Testing Accuracy: ", Ridge_reg_score)

    Lasso_reg = lm.Lasso()
    Lasso_reg.fit(X_train, y_train)
    Lasso_reg_score = Lasso_reg.score(X_test, y_test)
    print("Lasso Testing Accuracy: ", Lasso_reg_score)

    Elastic_reg = lm.ElasticNet()
    Elastic_reg.fit(X_train, y_train)
    Elastic_reg_score = Elastic_reg.score(X_test, y_test)
    print("Elastic Testing Accuracy: ", Elastic_reg_score)

    Huber_reg = lm.HuberRegressor()
    Huber_reg.fit(X_train, y_train)
    Huber_reg_score = Huber_reg.score(X_test, y_test)
    print("Huber Testing Accuracy: ", Huber_reg_score)

    Ransac_reg = lm.RANSACRegressor()
    Ransac_reg.fit(X_train, y_train)
    Ransac_reg_score = Ransac_reg.score(X_test, y_test)
    print("Ransac Testing Accuracy: ", Ransac_reg_score)

    Theil_reg = lm.TheilSenRegressor()
    Theil_reg.fit(X_train, y_train)
    Theil_reg_score = Theil_reg.score(X_test, y_test)
    print("Theil Testing Accuracy: ", Theil_reg_score)

    regressor = CatBoostRegressor()
    regressor.fit(X_train, y_train)
    regressor_score = regressor.score(X_test, y_test)
    print("CatBoost Testing Accuracy: ", regressor_score)

    models = [log_reg, lin_reg, Sgd_reg, Ridge_reg, Lasso_reg, Elastic_reg, Huber_reg, Ransac_reg, Theil_reg, regressor]
    scores = [log_reg_score, lin_reg_score, Sgd_reg_score, Ridge_reg_score, Lasso_reg_score, Elastic_reg_score, Huber_reg_score, Ransac_reg_score, Theil_reg_score, regressor_score]

    return models, scores


if __name__ == '__main__':
    data, test = load_data()
    X_train, y_train = Process_Data(data)
    X_test = Process_Test_Data(test)

    # Scaling the data using StandardScaler to normalize the data
    Scaler = StandardScaler()
    X_train = Scaler.fit_transform(X_train)
    X_test = Scaler.transform(X_test)

    models, scores = Machine_Learning(X_train, y_train)

    print("Models: ", models)
    print("Scores: ", scores)
    best_model = models[scores.index(max(scores))]
    print("Best Model: ", best_model)

    # Predicting the test data
    test_predictions = best_model.predict(X_test)
    print("Test Predictions: ", test_predictions)

    # Saving the predictions to a csv file
    # Only write the ID of teh test data and the predictions
    test['SalePrice'] = test_predictions
    test[['Id', 'SalePrice']].to_csv('Data/predictions.csv', index=False)
    print("Predictions saved to predictions.csv")
