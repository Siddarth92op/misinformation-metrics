# import pandas 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the Dataset as Data frame 
# Dataset 1 Variable Name: data1 

data1 = pd.read_csv("Dataset\\TruthfulQA.csv")
# Dataset 2 Variable Name: data2

data2 = pd.read_csv("Dataset\\ToxiFact\\datasets\\csv\\nr-ahr.csv")

# Dataset 3 Variable Name: data3 

data3 = pd.read_json("Dataset\\FActScore 2023\\unlabeled\\Alpaca-7B.jsonl", lines=True)


# print the statistics of the Datasets 

print("Statistics for Dataset 1:")
print(data1.describe())
print("\n")

print("Statistics for Dataset 2:")
print(data2.describe())
print("\n")

print("Statistics for Dataset 3:")
print(data3.describe())
print("\n")

# Data visualization 


