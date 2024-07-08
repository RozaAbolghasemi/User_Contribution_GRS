import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def CoalitionValue_Diversity(F, item_scores, p):
    """
    Calculate coalition value of a subset of users based on personalized item scores and the inverse of diversity.

    Parameters:
        F (set): A set of user IDs representing the subset of users.
        item_scores (dict): A dictionary where keys are user IDs and values are their corresponding personalized item scores.
        p (int): ID of the single user.

    Returns:
        float: CoalitionValue
    """
    n = len(item_scores)
    item_scores_array = np.array([item_scores[user] for user in range(1, n+1)]) # converting dict to array
    
    similarity_matrix = cosine_similarity(item_scores_array)
    
    X = F 
    i = p   
    Xi = F + [p]

    
    L_X = similarity_matrix[X][:, X]  
    L_Xi = similarity_matrix[Xi][:, Xi]
    
    L_X_inv = np.linalg.inv(L_X)    
    L_i_X = similarity_matrix[i, X]
    L_i_i = similarity_matrix[i, i]
    L_X_i = similarity_matrix[X, i] 


    ########### TWO ways to calculate diversity score: ################
    epsilon = 1e-100  # Small positive value to avoid taking log of non-positive values
    print("****", L_i_i - np.dot(L_i_X, np.dot(L_X_inv, L_X_i)), L_i_i - np.dot(L_i_X, np.dot(L_X_inv, L_i_X)))
    diversity1 = np.log(L_i_i - np.dot(L_i_X, np.dot(L_X_inv, L_X_i)))
    diversity = np.log(max(L_i_i - np.dot(L_i_X, np.dot(L_X_inv, L_i_X)), epsilon))
    
    # Calculate diversity as log det L_{X ∪ i} - log det L_{X}
    diversity2 = np.log(np.linalg.det(np.add(L_X, np.outer(L_i_X, L_i_X)))) - np.log(np.linalg.det(L_X))
    

    #CoalitionValue = 1- diversity2
    return diversity2
    
