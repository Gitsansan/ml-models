# Required imports

print("------Decision Tree------")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

print("\nReading in the data...")
my_data = pd.read_csv('datasets/drug200.csv')

print("Preprocessing the data...")
label_encoder = LabelEncoder()
sex_encode = label_encoder.fit(my_data['Sex'])
bp_encode = label_encoder.fit(my_data['BP'])
cho_encode = label_encoder.fit(my_data['Cholesterol'])



my_data['Sex'] = sex_encode.transform(my_data['Sex']) 
my_data['BP'] = bp_encode.transform(my_data['BP'])
my_data['Cholesterol'] = cho_encode.fit_transform(my_data['Cholesterol']) 

custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)

print("\nSplitting the data into training and testing sets...")
y = my_data['Drug']
X = my_data.drop(['Drug','Drug_num'], axis=1)

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)

print("Training the Decision Tree model...")
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

drugTree.fit(X_trainset,y_trainset)
print("\nDecision Tree has been trained.")

tree_predictions = drugTree.predict(X_testset)
print("Model Accuracy:", round(metrics.accuracy_score(y_testset, tree_predictions)*100,3), "%")

user_details = {}

print("\n--- User Input Section ---")
user_details['age'] = int(input("Enter Age (e.g., 23): "))
user_details['sex'] = input("Enter the sex(M/F): ").strip().upper()
user_details['bp'] = input("Enter the BP(LOW/NORMAL/HIGH): ").strip().upper()
user_details['cholesterol'] = input("Enter the cholesterol(HIGH/NORMAL): ").strip.upper()()
user_details['na_to_k'] = float(input("Enter the Na to K ratio (e.g., 15.0): "))

def encode_input(details):
    details['sex'] = sex_encode.transform(details['sex'])
    details['bp'] = bp_encode.transform(details['bp'])
    details['cholesterol'] = cho_encode.transform(details['cholesterol'])
    return details