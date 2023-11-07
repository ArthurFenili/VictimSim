import numpy as np
import skfuzzy as fuzz
from skfuzzy.cluster import cmeans
import pandas as pd

# Step 1: Prepare your dataset (assuming you have a DataFrame called 'data')
# Ensure your dataset is cleaned and contains relevant attributes

# Step 2: Define fuzzy sets and membership functions for your attributes
# You need to define linguistic variables, terms, and membership functions

# Step 3: Fuzzify your dataset
# Convert numerical attributes into fuzzy sets based on the membership functions

# Step 4: Apply Fuzzy C-Means (FCM) clustering
num_clusters = 3  # Adjust the number of clusters as needed
data_matrix = data.values.T  # Convert DataFrame to a data matrix

# Perform FCM clustering
cntr, u, u0, d, jm, p, fpc = cmeans(data_matrix, num_clusters, m=2, error=0.005, maxiter=1000, seed=0)

# Step 5: Generate rules based on cluster centers and membership degrees
for cluster_idx, cluster_center in enumerate(cntr):
    # Determine the condition part of the rule
    rule_condition = []
    for attr_idx, attr_values in enumerate(cluster_center):
        # Convert attribute values to fuzzy sets and get the membership degrees
        membership_degrees = fuzz.interp_membership(attribute_universe, attr_values, data_matrix[attr_idx])

        # Find the linguistic term with the highest membership degree
        term = linguistic_terms[np.argmax(membership_degrees)]

        # Add the condition to the rule
        rule_condition.append(f"{attribute_names[attr_idx]} is {term}")

    # Define the consequence part of the rule (e.g., output variable)

    # Create the rule
    rule = f"IF {' and '.join(rule_condition)} THEN {consequence_variable} is {cluster_idx}"

    # Print or store the rule
    print(rule)
