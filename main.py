import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
# lendo o dataset
tic_tact_data = pd.read_csv('tic-tac-toe.csv')
#convert all o values from the dataset to -1
tic_tact_data.replace(to_replace='o', value=-1, inplace=True)
tic_tact_data.replace(to_replace='b', value=0, inplace=True)
tic_tact_data.replace(to_replace='x', value=1, inplace=True)
tic_tact_data.replace(to_replace='negativo', value=-1, inplace=True)
tic_tact_data.replace(to_replace='positivo', value=1, inplace=True)
print(tic_tact_data.head())
# salvando o dataset modificado
tic_tact_data.to_csv('tic-tact-data2222.csv', index=False)
# separando dataset
X = tic_tact_data.iloc[:, :-1].values
y = tic_tact_data.iloc[:, -1].values
# separando dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# criando o classificador
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
# treinando o classificador
classifier.fit(X_train, y_train)
# previsao
y_pred = classifier.predict(X_test)

# calculando a acuracia
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
# plotando o grafico
sns.heatmap(tic_tact_data.corr(), annot=True)
plt.show()


user_input = input("Digite 9 valores separados por espaço: ")

user_input = user_input.replace(',', ' ')
user_input = user_input.split()
user_input = [float(i) for i in user_input]
tic_tact_predictions = classifier.predict([user_input])
print(tic_tact_predictions)
#print X venceu if tic_tact_predictions == 1 else print O venceu if tic_tact_predictions == -1 else print empate
if tic_tact_predictions == 1:
    print("X venceu")
else: print("X não venceu")