import sklearn.linear_model as lm
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
import tensorflow as tf

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class MyModel:
    def __init__(self, input_shape):
        super().__init__()
        # Create the model layers
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, input_shape=input_shape),
            tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(1, activation=tf.nn.leaky_relu)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                             beta_1=0.9,
                                             beta_2=0.999,
                                             epsilon=1e-07,
                                             amsgrad=False)

        self.model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

        # Model checkpoint callback to save the best model based on lowest MAE
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5',
                                                             monitor='mae',
                                                             verbose=1,
                                                             save_best_only=True,
                                                             mode='min')

    def train(self, X_train, y_train, epochs=1000, batch_size=32):

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 callbacks=[self.checkpoint])
        return history

    def evaluate(self, X_test, y_test):
        loss, mae = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")

    def predict(self, X_new):
        return self.model.predict(X_new)

    def load_best_model(self):
        # Loads the best model saved during training based on lowest validation MAE.
        self.model = tf.keras.models.load_model('best_model.h5')


def load_data():
    data = pd.read_csv('Data/train.csv')
    test = pd.read_csv('Data/test.csv')
    return data, test


# Changing the data type of the columns to category and then to numerical values for the model
def data_categories(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].astype('category')
            data[column] = data[column].cat.codes

    return data


def Process_Data(data):
    data = data_categories(data)

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
    test = data_categories(test)

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

    models = [log_reg, lin_reg, Sgd_reg, Ridge_reg, Lasso_reg, Elastic_reg, Huber_reg, Ransac_reg, Theil_reg]
    scores = [log_reg_score, lin_reg_score, Sgd_reg_score, Ridge_reg_score, Lasso_reg_score, Elastic_reg_score, Huber_reg_score, Ransac_reg_score, Theil_reg_score]

    return models, scores

#
# return models, scores


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

    print(X_test.shape)
    test_predictions = []
    print("Predicting the test data")
    for i in range(X_test.shape[0]):
        prediction = best_model.predict(X_test[i:i + 1])[0]
        test_predictions.append(prediction)
    print("Test Predictions: ", test_predictions)

    test['SalePrice'] = test_predictions
    test[['Id', 'SalePrice']].to_csv('Data/predictions_myModel.csv', index=False)
    print("Predictions saved to predictions.csv")

    model = MyModel(input_shape=(X_train.shape[1],))  # Pass the input shape
    model.train(X_train, y_train, epochs=1000, batch_size=32)
    test_predictions = model.predict(X_test)

    # Saving the predictions to a csv file
    # Only write the ID of teh test data and the predictions
    test['SalePrice'] = test_predictions
    test[['Id', 'SalePrice']].to_csv('Data/predictions_myModel.csv', index=False)
    print("Predictions saved to predictions_myModel.csv")
