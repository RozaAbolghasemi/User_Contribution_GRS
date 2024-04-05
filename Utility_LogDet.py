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
    
    X = list(F)  # Convert set to list for indexing
    i = p - 1  # Adjusting user ID to index (0-based index)
    
    L_X = similarity_matrix[X][:, X]
    L_X_inv = np.linalg.inv(L_X)
    
    L_i_X = similarity_matrix[i, X]
    L_i_i = similarity_matrix[i, i]
    
    diversity = np.log(L_i_i - np.dot(L_i_X, np.dot(L_X_inv, L_i_X)))
    
    return CoalitionValue
    
