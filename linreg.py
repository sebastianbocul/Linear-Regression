import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#wczytanie danych
boston_dataset = load_boston()
boston_dataset.keys()

#wczytanie danych do Pandy
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()

#stworzenie kolumny MEDV
boston['MEDV'] = boston_dataset.target


#Sprawdzenie danych
print(boston.isnull().sum())

#wizalizacja danych(wykres)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()

#koleracja danych
correlation_matrix = boston.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)


#stowrzenie wykresów MEDV-LSTAT oraz MEDV-RM
plt.figure(figsize=(20, 5))
features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
    plt.show()
    
plt.close()

#przygotowanie danych
X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']

#Podzielenie danych na testowe i treningowe
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)


#trenowanie modelu
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


#ewaluacja danych treningowych
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("Wydajnosc modelu dla danych trenujacych")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

#ewaluacja danych testowych
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("Wydajnosc modelu dla danych testowych")
print("--------------------------------------")
print('RMSE: {}'.format(rmse))
print('R2 score:{}'.format(r2))

#rysowanie modelu
plt.close()
plt.scatter(Y_test, y_test_predict)
plt.show()
