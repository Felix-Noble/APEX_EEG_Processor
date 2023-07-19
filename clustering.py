import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
from APEX_EEG_Processor import EEG_Processor
from scipy.spatial import distance
import json, pickle

# Function to plot average distance of clusters for each condition
def plot_clusters(clustered_data):
    # Creating a new figure
    plt.figure(figsize=(10, 5))

    # Loop over all conditions in the clustered data
    for i, (condition, clusters) in enumerate(clustered_data.items()):

        # Get the cluster numbers and average distances
        cluster_numbers = list(clusters.keys())
        average_distances = [avg_distance for _, avg_distance in clusters.values()]

        # Plot the average distances
        plt.plot(cluster_numbers, average_distances, label=f'Condition {condition}')

    # Adding title and labels
    plt.title('Average Distance of Clusters for Each Condition')
    plt.xlabel('Cluster Number')
    plt.ylabel('Average Distance')
    plt.legend()

    # Display the plot
    plt.show()

def organise_files(filenames):
    files_dict = {}
    for filename in filenames:
        f = AP.get_main_filename(filename)
        f = f.split('-')
        # The condition name starts from the second part of the filename
        condition = "-".join(f[1:])  # Skip subject ID
        if condition not in files_dict:
            # If this is the first time we've seen this condition, initialize the list
            files_dict[condition] = []
        # Add the full filename to the list of files for this condition if it's not already there
        if filename not in files_dict[condition]:
            files_dict[condition].append(filename)
    return files_dict


# Load the EEG data from CSV files into a list of Pandas DataFrames

data = json.load(open('analysis.json', 'r'))

conditions = ['CR-lvl grr14-AVG', 'CR-lvl lss13-AVG',
              'CR-trial-first_half-AVG', 'CR-trial-second_half-AVG',
             'CRCT-AVG', 'MISS-AVG',
             'MS-lvl grr14-AVG', 'MS-lvl lss13-AVG',
             'MS-trial-first_half-AVG', 'MS-trial-second_half-AVG']

clustered_data = {}

for condition in conditions:
    values = data[condition]

    # Convert the list of lists to a numpy array
    dissimilarity_values = np.array(values)

    # Compute the linkage matrix for the dissimilarity values
    Z = linkage(pdist(dissimilarity_values), method='ward')

    clusters = {}

    # Loop over all unique distance thresholds in the linkage matrix
    for t in np.unique(Z[:, 2]):

        # Get cluster assignments based on the distance threshold
        cluster_assignments = fcluster(Z, t, criterion='distance')

        # Loop over all unique cluster assignments
        for cluster_number in np.unique(cluster_assignments):
            # Get the data points for this cluster
            cluster_data = dissimilarity_values[cluster_assignments == cluster_number]

            # Compute the average distance for the cluster data
            average_distance = np.mean(cluster_data)

            # Store the cluster data and average distance in the dictionary
            clusters[cluster_number] = (cluster_data, average_distance)

    # Now `clusters` is a dictionary where the keys are cluster numbers and the values are tuples
    # (cluster_data, average_distance), where `cluster_data` is the data points in the cluster and
    # `average_distance` is their average distance.

    # Add the clustered data for this condition to the `clustered_data` dictionary
    clustered_data[condition] = clusters

with open('clusters.pkl', 'w')as file:
    pickle.dump(clustered_data)

plot_clusters(clustered_data)
#
# # Reshape the ERP data to match the input format required for clustering
# X = np.stack(erp_data)
# # Calculate the linkage matrix using hierarchical clustering
# Z = linkage(X, method='average', metric='euclidean')
# filenames = [AP.get_main_filename(file) for file in cond_files]
#
# # Plot the dendrogram
# plt.figure(figsize=(10, 6))
# dendrogram(Z, labels=filenames, leaf_rotation=90)
# plt.xlabel('Data Files')
# plt.ylabel('Distance')
# plt.title('Hierarchical Clustering Dendrogram')
# plt.show()
#
# # Determine the stable periods based on the generated dendrogram
# # Define a threshold distance to determine the stability
# threshold = 0.1  # Adjust the threshold value as needed
#
# # Extract the clusters based on the threshold distance
# clusters = []
# for t in np.unique(Z[:, 2]):
#     cluster = Z[Z[:, 2] == t, :2].flatten().astype(int)
#     if len(cluster) > 1:
#         clusters.append(cluster)
#
# # Identify stable periods within and between conditions
# stable_periods = []
# for cluster in clusters:
#     cluster_data = X[cluster]
#     mean_cluster = np.mean(cluster_data, axis=0)
#     distances = distance.cdist(cluster_data, [mean_cluster], metric='correlation')
#     mean_distance = np.mean(distances)
#     if mean_distance <= threshold:
#         stable_periods.append(cluster)
#
# # Print the stable periods
# for i, stable_period in enumerate(stable_periods):
#     print(f'Stable Period {i + 1}: {data_files[stable_period]}')

#######
# read = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\epochs as csv"
# read_fif = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\cropped_files"
# write = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\write"
# AP = EEG_Processor(read, write, mne_log=False)
# data_files = AP.subj_csv_files
# sorted_files = organise_files(data_files)
# conds = ['CR-lvl grr14-AVG', 'CR-lvl lss13-AVG', 'CR-trial-first_half-AVG', 'CR-trial-second_half-AVG', 'CRCT-AVG', 'MISS-AVG', 'MS-lvl grr14-AVG', 'MS-lvl lss13-AVG', 'MS-trial-first_half-AVG', 'MS-trial-second_half-AVG']
#
# selected_condition = conds[0]
# cond_files = sorted_files[selected_condition]
# evoked_data = [pd.read_csv(file) for file in cond_files]
#
# # Extract the ERP data from the Pandas DataFrames
# erp_data = [df.values.flatten() for df in evoked_data]
