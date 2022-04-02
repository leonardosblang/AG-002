import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import datetime
import random
import warnings
import pickle
import json
import re
import sklearn
from sklearn.metrics import confusion_matrix


warnings.filterwarnings('ignore')


# lendo o dataset

tic_tact_data = pd.read_csv('tic-tac-toe.csv')
#convert all o values from the dataset to -1

tic_tact_data.replace(to_replace='o', value=-1, inplace=True)
tic_tact_data.replace(to_replace='b', value=-0, inplace=True)
tic_tact_data.replace(to_replace='x', value=-1, inplace=True)
tic_tact_data.replace(to_replace='negativo', value=-1, inplace=True)
tic_tact_data.replace(to_replace='positivo', value=1, inplace=True)

print(tic_tact_data.head())
# salvando o dataset modificado
tic_tact_data.to_csv('tic-tact-data2222.csv', index=False)

# separando dataset

train_data = tic_tact_data.sample(frac=0.8, random_state=200)
test_data = tic_tact_data.drop(train_data.index)


#treinando com nearest neighbor

from sklearn.neighbors import KNeighborsClassifier

tic_tact_classifier = KNeighborsClassifier(n_neighbors=3)
tic_tact_classifier.fit(train_data.iloc[:,0:9], train_data.iloc[:,9])

#prevendo os valores
tic_tact_predictions = tic_tact_classifier.predict(test_data.iloc[:,0:9])

#classificando o modelo
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_data.iloc[:,9], tic_tact_predictions))
print(classification_report(test_data.iloc[:,9], tic_tact_predictions))

#entrando com valores do usuario

user_input = input("Digite 9 valores separados por espaço: ")
#remove the , from the input
user_input = user_input.replace(',', ' ')
user_input = user_input.split()
user_input = [float(i) for i in user_input]


tic_tact_predictions = tic_tact_classifier.predict([user_input])
print(tic_tact_predictions)







