import pandas as pd
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity


def CarData_preprocessing():
    User_num = 60
    Item_num = 10

    ComparisonMatrix = np.zeros((User_num, Item_num, Item_num))

    filename = "./Data/CarDataset/prefs1.csv"
    # Open the CSV file and read line by line
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            UserID = int(row[0]) - 1
            Item1ID = int(row[1]) - 1
            Item2ID = int(row[2]) - 1
            ComparisonMatrix[UserID][Item1ID][Item2ID] = 1
            ComparisonMatrix[UserID][Item2ID][Item1ID] = 0

    filename = "./Data/CarDataset/prefs1.csv"
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["User_ID", "Item1_ID", "Item2_ID", "PairwiseScore"]
        writer.writerow(header)

        for user in range(User_num):
            for item1 in range(Item_num):
                for item2 in range(item1 + 1, Item_num):
                    List = [user + 1, item1 + 1, item2 + 1, ComparisonMatrix[user][item1][item2]]
                    writer.writerow(List)

    PairwiseRates = pd.read_csv("./Data/CarDataset/prefs1.csv")  # .to_numpy()
    # Item_pairs is a list containing paired items
    ii, iii = PairwiseRates.shape
    Item_pairs = []
    for i in range(ii):
        Item_pairs.append((PairwiseRates.Item1_ID[i], PairwiseRates.Item2_ID[i]))
    unique_Item_pairs = list(dict.fromkeys(Item_pairs))
    print("*")

    # To make access to indexes easier, dictionaries are used:
    UniqueUserIDs = np.unique(PairwiseRates.User_ID)
    unique_Item_pairs
    dict_index2user = {}
    keys = range(len(UniqueUserIDs))
    values = UniqueUserIDs
    for i in keys:
        dict_index2user[values[i]] = i

    dict_index2itempairs = {}
    keys = range(len(unique_Item_pairs))
    values = unique_Item_pairs
    for i in keys:
        dict_index2itempairs[values[i]] = i
    print("**")

    # Mtrix R:   Rows:unique users   Columns: Unique Item pairs
    R = np.ones([len(UniqueUserIDs), len(unique_Item_pairs)])
    for row in range(len(PairwiseRates)):
        u = dict_index2user[PairwiseRates.User_ID[row]]
        i = dict_index2itempairs[(
            PairwiseRates.Item1_ID[row], PairwiseRates.Item2_ID[row])]
        R[u][i] = PairwiseRates.PairwiseScore[row]

    return R


def FoodData_preprocessing():
    User_num = 20
    Item_num = 6
    ComparisonMatrix = np.zeros((User_num, Item_num, Item_num))
    ComparisonMatrix[0][:][:] = pd.read_csv("./FoodData/User1.csv", index_col=0).to_numpy()
    ComparisonMatrix[1][:][:] = pd.read_csv("./FoodData/User2.csv", index_col=0).to_numpy()
    ComparisonMatrix[2][:][:] = pd.read_csv("./FoodData/User3.csv",index_col=0).to_numpy()
    ComparisonMatrix[3][:][:] = pd.read_csv("./FoodData/User4.csv", index_col=0).to_numpy()
    ComparisonMatrix[4][:][:] = pd.read_csv("./FoodData/User5.csv", index_col=0).to_numpy()

    ComparisonMatrix[5][:][:] = pd.read_csv("./FoodData/User6.csv", index_col=0).to_numpy()
    ComparisonMatrix[6][:][:] = pd.read_csv("./FoodData/User7.csv", index_col=0).to_numpy()
    ComparisonMatrix[7][:][:] = pd.read_csv("./FoodData/User8.csv",index_col=0).to_numpy()
    ComparisonMatrix[8][:][:] = pd.read_csv("./FoodData/User9.csv", index_col=0).to_numpy()
    ComparisonMatrix[9][:][:] = pd.read_csv("./FoodData/User10.csv", index_col=0).to_numpy()

    ComparisonMatrix[10][:][:] = pd.read_csv("./FoodData/User11.csv", index_col=0).to_numpy()
    ComparisonMatrix[11][:][:] = pd.read_csv("./FoodData/User12.csv", index_col=0).to_numpy()
    ComparisonMatrix[12][:][:] = pd.read_csv("./FoodData/User13.csv",index_col=0).to_numpy()
    ComparisonMatrix[13][:][:] = pd.read_csv("./FoodData/User14.csv", index_col=0).to_numpy()
    ComparisonMatrix[14][:][:] = pd.read_csv("./FoodData/User15.csv", index_col=0).to_numpy()

    ComparisonMatrix[15][:][:] = pd.read_csv("./FoodData/User16.csv",index_col=0).to_numpy()
    ComparisonMatrix[16][:][:] = pd.read_csv("./FoodData/User17.csv", index_col=0).to_numpy()
    ComparisonMatrix[17][:][:] = pd.read_csv("./FoodData/User.csv",index_col=0).to_numpy()
    ComparisonMatrix[18][:][:] = pd.read_csv("./FoodData/User18.csv",index_col=0).to_numpy()
    ComparisonMatrix[19][:][:] = pd.read_csv("./FoodData/User19.csv", index_col=0).to_numpy()

    filename = "./Data/Data_Pedro/PairwiseRates_Food.csv"
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["User_ID", "Item1_ID", "Item2_ID", "PairwiseScore"]
        writer.writerow(header)

        for user in range(User_num):
            for item1 in range(Item_num):
                for item2 in range(item1 + 1, Item_num):
                    List = [user + 1, item1 + 1, item2 + 1, ComparisonMatrix[user][item1][item2]]
                    writer.writerow(List)

    PairwiseRates = pd.read_csv("./Data/Data_Pedro/PairwiseRates_Food.csv")  # .to_numpy()
    # Item_pairs is a list containing paired items
    ii, iii = PairwiseRates.shape
    Item_pairs = []
    for i in range(ii):
        Item_pairs.append((PairwiseRates.Item1_ID[i], PairwiseRates.Item2_ID[i]))
    unique_Item_pairs = list(dict.fromkeys(Item_pairs))
    print("*")

    # To make access to indexes easier, dictionaries are used:
    UniqueUserIDs = np.unique(PairwiseRates.User_ID)
    unique_Item_pairs
    dict_index2user = {}
    keys = range(len(UniqueUserIDs))
    values = UniqueUserIDs
    for i in keys:
        dict_index2user[values[i]] = i

    dict_index2itempairs = {}
    keys = range(len(unique_Item_pairs))
    values = unique_Item_pairs
    for i in keys:
        dict_index2itempairs[values[i]] = i
    print("**")

    # Mtrix R:   Rows:unique users   Columns: Unique Item pairs
    R = np.ones([len(UniqueUserIDs), len(unique_Item_pairs)])
    for row in range(len(PairwiseRates)):
        u = dict_index2user[PairwiseRates.User_ID[row]]
        i = dict_index2itempairs[(
            PairwiseRates.Item1_ID[row], PairwiseRates.Item2_ID[row])]
        R[u][i] = PairwiseRates.PairwiseScore[row]

    return R


# Matrix Factorization
def mf(R, k, n_epoch=5000, lr=.0003, l2=.04):  # n_epoch=5000, lr=.0003
    print("Rnning Matrix Factorization....")
    tol = .001  # Tolerant loss.
    m, n = R.shape
    R2 = R - 10
    # Initialize the embedding weights.
    P = np.random.rand(m, k)
    Q = np.random.rand(n, k)
    for epoch in range(n_epoch):
        # Update weights by gradients.
        for u, i in zip(*R2.nonzero()):
            err_ui = R[u, i] - P[u, :].dot(Q[i, :])
            for j in range(k):
                P[u][j] += lr * (2 * err_ui * Q[i][j] - l2 / 2 * P[u][j])
                Q[i][j] += lr * (2 * err_ui * P[u][j] - l2 / 2 * Q[i][j])
        # compute the loss.
        E = (R - P.dot(Q.T)) ** 2
        obj = E[R.nonzero()].sum() + lr * ((P ** 2).sum() + (Q ** 2).sum())
        if obj < tol:
            break

        # Saving the embeddings:
        pd.DataFrame(P).to_csv('./Data/Data_Pedro/User_Embedding.csv', header=True, index=True)
        pd.DataFrame(Q).to_csv('./Data/Data_Pedro/PairItems_Embedding.csv', header=True, index=True)
    return P, Q


# Clustering with Shapley value
def calculate_diversity(User_Embedding, X):
    """
    Calculate the diversity of each user not in the cluster X
    based on the cosine similarity kernel matrix.

    Parameters:
    - User_Embedding: 2D array containing user vectors
    - X: Current cluster

    Returns:
    - diversity: Array containing the diversity values for each user not in X
    """
    n = User_Embedding.shape[0]
    similarity_matrix = cosine_similarity(User_Embedding)
    L_X = similarity_matrix[X][:, X]
    L_X_inv = np.linalg.inv(L_X)
    diversity = np.zeros(n)
    for i in range(n):
        if i not in X:
            L_i_X = similarity_matrix[i][X]
            L_i_i = similarity_matrix[i][i]
            diversity[i] = np.log(L_i_i - np.dot(L_i_X, np.dot(L_X_inv, L_i_X)))
    return diversity


def cluster_users(User_Embedding, threshold=0, max_users=None):
    """
    Cluster users based on the inverse of diversity in a Determinantal Point Process (DPP) kernel matrix.

    Parameters:
    - User_Embedding: 2D array containing user vectors
    - threshold: Diversity threshold to stop clustering (default: 0)
    - max_users: Maximum number of users in a cluster (default: None)

    Returns:
    - clusters: List of clusters, where each cluster is represented as a list of user indices
    """
    n = User_Embedding.shape[0]
    clusters = []
    remaining_users = list(range(n))

    while remaining_users:
        current_cluster = []
        # Calculate diversity for each user not in any cluster
        diversity = calculate_diversity(User_Embedding, [])
        # Find the user with the least diversity and add it to a new cluster
        next_user_idx = np.argmin(diversity[remaining_users])
        next_user = remaining_users[next_user_idx]
        current_cluster.append(next_user)
        remaining_users.pop(next_user_idx)

        while True:
            # Calculate diversity for each user not in the current cluster
            diversity = calculate_diversity(User_Embedding, current_cluster)
            # Check if remaining_users is empty before finding next_user
            if remaining_users:
                next_user_idx = np.argmin(diversity[remaining_users])
                next_user = remaining_users[next_user_idx]
                remaining_users.pop(next_user_idx)
            else:
                break
            # Add the next_user to the current cluster
            current_cluster.append(next_user)
            # Check stopping conditions
            print(diversity[next_user])
            if (diversity[next_user] >= threshold) or \
                    (max_users is not None and len(current_cluster) >= max_users):
                clusters.append(current_cluster)
                break
    return clusters













