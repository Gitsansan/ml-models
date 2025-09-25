# Required importrs

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

my_data = pd.read_csv('datasets/drug200.csv')

print(my_data.head())