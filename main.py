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


# read the tic-tact dataset
tic_tact_data = pd.read_csv('tic-tac-toe.csv')
#convert all o values from the dataset to -1
tic_tact_data.replace(to_replace='o', value=-1, inplace=True)
tic_tact_data.replace(to_replace='b', value=-0, inplace=True)
tic_tact_data.replace(to_replace='x', value=-1, inplace=True)
tic_tact_data.replace(to_replace='negativo', value=-1, inplace=True)
tic_tact_data.replace(to_replace='positivo', value=1, inplace=True)
print(tic_tact_data.head())

