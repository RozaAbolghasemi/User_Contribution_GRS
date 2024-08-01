import pandas as pd
import numpy as np
import csv
from utils import *
import timeit




# Reading data:
dataset_choice = input("Please indicate which dataset you choose? Enter 1 for Food and 2 for Car Dataset: ")
start = timeit.default_timer()

if dataset_choice == "1":
    R = FoodData_preprocessing()
    file_path = "group_Mambers_IDs_FoodData_Diversity_Clustering.csv"

if dataset_choice == "2":
    R = CarData_preprocessing()
    file_path = "group_Ma1mbers_IDs_CarData_Diversity_Clusteringlustering.csv"
print (R)

# matrix factorization:
k = 10  # len of embeddings
User_Embedding, PairItems_Embedding = mf(R, k, n_epoch=100, lr=.01, l2=.04) # Matrix Factorization


# Clustering with Shapley value:
clusters = cluster_users(User_Embedding, threshold=1.0, max_users=4)
print("Clusters:", clusters)

# Saving the groups ids as a csv file:



# Write the 2D list to the CSV file
with open(file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    for row in clusters:
        csv_writer.writerow(row)

print("CSV file saved successfully.")

#Calculate the "Diversity clustering Execution time"
stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed for Diversity clustering: "+str(execution_time)) # It returns time in seconds
