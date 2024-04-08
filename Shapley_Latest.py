from itertools import combinations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
num_top_items=2
item_score = {
'User1': [.34, .33, .54, .36, .67, .77], 'User2': [.51, .45, .55, .42, .66, .42],
               'User3': [.47, .52, .49, .53, .47, .53], 'User4': [.71, .62, .57, .44, .35, .32],
               'User5': [.35, .74, .41, .49, .55, .46],'User6': [.23, .79, .44, .53, .65, .66], 'User7': [.20, .52, .31, .55, .76, .52],
               'User8': [.45, .48, .55, .64, .47, .41], 'User9': [.35, .34, .46, .57, .61, .69],
               'User10': [.32, .60, .39, .75, .51, .44],'User11': [.38, .34, .47, .57, .63, .61], 'User12': [.48, .48, .52, .39, .67, .46],
                   'User13': [.43, .55, .46, .48, .52, .56], 'User14': [.68, .62, .53, .42, .38, .37],
                   'User15': [.37, .67, .43, .47, .79, .53], 'User16': [.27, .47, .43, .54, .64, .65], 'User17': [.28, .54, .37, .56, .89, .51],
                   'User18': [.41, .51, .51, .60, .52, .47], 'User19': [.41, .37, .47, .58, .60, .66],
                   'User20': [.29, .55, .43, .67, .59, .47]}

def CoalitionValue_Diversity(F, item_scores, p):
    """
    Calculate coalition value of a user based on personalized item scores and the inverse of diversity.

    Parameters:
        F (set): A set of user IDs representing the subset of users.
        item_scores (dict): A dictionary where keys are user IDs and values are their corresponding personalized item scores.
        p (int): ID of the single user.

    Returns:
        float: CoalitionValue
    """
    n = len(item_scores)

    item_scores_array = np.array([item_scores[user] for user in item_scores])  # converting dict to array
    similarity_matrix = cosine_similarity(item_scores_array)

    X = list(F)  # Convert set to list for indexing

    # Extract the numerical part from the user ID string
    user_id = int(p[4:])
    i = user_id - 1  # Adjust user ID to index (0-based index)


    # Correctly index the similarity matrix using the integer index
    L_X = similarity_matrix[
        np.ix_([int(u[4:]) - 1 for u in X], [int(u[4:]) - 1 for u in X])]  # Convert user IDs to integer indices


    # If L_X is not square, pad it with zeros to make it square
    if L_X.shape[0] != L_X.shape[1]:
        max_dim = max(L_X.shape)
        L_X_padded = np.zeros((max_dim, max_dim))
        L_X_padded[:L_X.shape[0], :L_X.shape[1]] = L_X
        L_X = L_X_padded

    L_X_inv = np.linalg.inv(L_X)

    L_i_X = similarity_matrix[i, [int(u[4:]) - 1 for u in X]]
    L_i_i = similarity_matrix[i, i]

    diversity = np.log(L_i_i - np.dot(L_i_X, np.dot(L_X_inv, L_i_X)))


    return diversity


def calculate_shapley_values(user_ratings, ground_truth_group_ratings, group_names):
    shapley_values = {}

    for group_name in group_names:
        shapley_values[group_name] = {}
        users = list(user_ratings[group_name].keys())
        for user in users:
            #print('user',user)
            shapley_values[group_name][user] = calculate_shapley_value(
                user, users, user_ratings[group_name], ground_truth_group_ratings, group_name
            )

    return shapley_values


def calculate_shapley_value(user, users, user_ratings, ground_truth_ratings, group_name):
    shapley_value = 0
    subset_users = set(users)
    subset_users.remove(user)  # Exclude the current user from the coalition
    #print('subset_users',subset_users)
    complement_subsets = generate_all_subsets(subset_users)
    #print('complement_subsets',complement_subsets)
    l = len(complement_subsets)

    for complement_subset in complement_subsets:
        full_subset = set(complement_subset)
        full_subset.add(user)  # Include the current user in the coalition
        #print('full_subset',full_subset)
        #print('complement_subset',complement_subset)
        #print('l',l)
        #coalition_value = CoalitionValue_Diversity(full_subset, item_score, user)
        shapley_value = CoalitionValue_Diversity(complement_subset, item_score, user)
        #shapley_value += coalition_value - complement_value

    shapley_value /= l
    #print(shapley_value)
    return shapley_value



def combine_user_ratings(user_ratings):
    return np.mean(list(user_ratings.values()), axis=0)

def generate_all_subsets(users):
    all_subsets = []
    for i in range(1, len(users) + 1):
        all_subsets.extend(combinations(users, i))
    return all_subsets

def normalize_shapley_values(shapley_values):
    normalized_shapley_values = {}
    for group_name, group_shapley in shapley_values.items():
        max_shapley = max(group_shapley.values())
        min_shapley = min(group_shapley.values())
        normalized_shapley_values[group_name] = {user: (shapley - min_shapley) / (max_shapley - min_shapley) for user, shapley in group_shapley.items()}
        #print(normalized_shapley_values)
    return normalized_shapley_values


def calculate_group_ratings(user_ratings, shapley_values,group_names,ground_truth_group_ratings):
    group_ratings = {group_name: np.zeros(len(ground_truth_group_ratings[group_name])) for group_name in group_names}


    for group_name in group_names:
        for item in range(len(group_ratings[group_name])):
            for user in user_ratings[group_name]:
                shapley_value = shapley_values[group_name][user]
                group_ratings[group_name][item] += shapley_value * user_ratings[group_name][user][item]
    #print(group_ratings)
    return group_ratings

def find_top_items(group_ratings, num_top_items=2):
    top_items = {group_name: [] for group_name in group_ratings}

    for group_name in group_ratings:
        item_ratings = [(item, rating) for item, rating in enumerate(group_ratings[group_name])]
        sorted_item_ratings = sorted(item_ratings, key=lambda x: x[1], reverse=True)
        top_items[group_name] = [item for item, _ in sorted_item_ratings[:num_top_items]]

    return top_items

def calculate_user_satisfaction(group_name, top_items, user_ratings, threshold):
    top_items_for_group = top_items[group_name]
    satisfaction_values = []
    total_users_in_group = len(user_ratings[group_name])

    for item_idx in top_items_for_group:
        satisfied_count = sum(
            1 for user in user_ratings[group_name] if user_ratings[group_name][user][item_idx] >= threshold)

        satisfaction_fraction = satisfied_count / total_users_in_group

        satisfaction_values.append(satisfaction_fraction)

    return satisfaction_values
def calculate_precision(group_name, top_items, num_top_items,user_ratings, threshold):
    top_items_for_group = top_items[group_name]
    precision = []
    total_users_in_group = len(user_ratings[group_name])

    precision_count = 0  # Initialize precision_count outside the loop

    for item_idx in top_items_for_group:
        precision_count += sum(
            1 for user in user_ratings[group_name] if user_ratings[group_name][user][item_idx] >= threshold)


    precision_fraction = precision_count / (total_users_in_group * num_top_items)


    return precision_fraction

def calculate_precision1(ground_truth_group_ratings, top_items, group_names, num_top_items=2, threshold=0.4):
    y_true = []
    y_pred = []

    for group_name in group_names:
        # Convert ground truth ratings to a NumPy array
        ground_truth_ratings = np.array(ground_truth_group_ratings[group_name])

        # Threshold ground truth ratings for the top items
        y_true.extend((ground_truth_ratings[:num_top_items] > threshold).astype(int))

        # Convert top_items_group to a NumPy array
        top_items_group = np.array(top_items[group_name])

        # Threshold predicted ratings for the top items
        y_pred.extend((top_items_group[:num_top_items] > threshold).astype(int))

    # Convert the lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("y_true shape:", y_true.shape)
    print("y_pred shape:", y_pred.shape)

    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall,f1,accuracy






group_names = ['Group5', 'Group6', 'Group7']
user_ratings= {'Group5': {
            'User3': [.47, .52, .49, .53, .47, .53],
            'User4': [.71, .62, .57, .44, .35, .32],
            'User8': [.45, .48, .55, .64, .47, .41],
            'User9': [.35, .34, .46, .57, .61, .69],
            'User13': [.43, .55, .46, .48, .52, .56],
            'User14': [.68, .62, .53, .42, .38, .37],
            'User15': [.37, .67, .43, .47, .79, .53],
            'User17': [.28, .54, .37, .56, .89, .51],
            'User18': [.41, .51, .51, .60, .52, .47],
            'User19': [.41, .37, .47, .58, .60, .66]},
        'Group6': {'User1': [.34, .33, .54, .36, .67, .77],
                   'User10': [.32, .60, .39, .75, .51, .44],
                   'User12': [.48, .48, .52, .39, .67, .46],
                   'User16': [.27, .47, .43, .54, .64, .65],
                   'User20': [.29, .55, .43, .67, .59, .47]},
        'Group7': {'User11': [.38, .34, .47, .57, .63, .61],
                   'User6': [.23, .79, .44, .53, .65, .66]},}
ground_truth_group_ratings= {'Group5': [0.45600000000000007, 0.522, 0.4840000000000001, 0.5289999999999999, 0.5599999999999999, 0.505],
        'Group6': [0.34, 0.48599999999999993, 0.462, 0.542, 0.616, 0.558],
        'Group7': [0.305, 0.5650000000000001, 0.45499999999999996, 0.55, 0.64, 0.635],
        }


start_time = time.time()
# Calculate Shapley values
shapley_values = calculate_shapley_values(user_ratings, ground_truth_group_ratings, group_names)
# Normalize Shapley values
normalized_shapley_values = normalize_shapley_values(shapley_values)

# Use normalized Shapley values in further calculations
group_ratings = calculate_group_ratings(user_ratings, normalized_shapley_values, group_names, ground_truth_group_ratings)
end_time = time.time()
 # Calculate the elapsed time
elapsed_time = end_time - start_time
# Print the elapsed time
print(f"Execution time: {elapsed_time} seconds")
# Print group names and ratings
for group_name, ratings in group_ratings.items():
    print(f"{group_name}: {ratings}")

# Find the top items
top_items = find_top_items(group_ratings, num_top_items)

# Print the top items for each group
for group_name, items in top_items.items():
    print(f"Top {num_top_items} Priority Items for {group_name}:")
    for item in items:
        print(f"Item {item + 1}")

# Calculate and print user satisfaction for each group
threshold = 0.4  # Set your desired threshold value
total_weighted_satisfaction = 0
total_weight = 0

for group_name in group_names:
    satisfaction_values = calculate_user_satisfaction(group_name, top_items, user_ratings, threshold)
    total_satisfaction = sum(satisfaction_values) / len(top_items[group_name])
    total_weighted_satisfaction += total_satisfaction * len(top_items[group_name])
    total_weight += len(top_items[group_name])
    print(f"Total User Satisfaction for Group {group_name}: {total_satisfaction * 100:.2f}%")

# Calculate overall user satisfaction
overall_satisfaction = total_weighted_satisfaction / total_weight
print(f"\nOverall User Satisfaction: {overall_satisfaction * 100:.2f}%")



# precision_dict = GR.calculate_precision1(ground_truth_group_ratings, top_items, group_names, num_top_items=2, threshold=0.6)
precision,recall,f1,accuracy= calculate_precision1(ground_truth_group_ratings, top_items, group_names)
print("Precision:", precision)
print("recall:", recall)
print("f1:", f1)
print("accuracy:", accuracy)





