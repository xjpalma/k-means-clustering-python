import numpy as np
from Clustering import KMeans


def load_data():
    """Method to load the iris data set from CSV file"""
    # Path to CSV file
    file_path = "data/data.csv"
    # Load data using pandas
    data_points = np.loadtxt(
        file_path,
        delimiter=",",
        skiprows=1,
        usecols=(2, 3),
    )

    # Return the required data in the form of a list as per code requirements in other methods
    return data_points.tolist()


if __name__ == "__main__":
    # Load dataset
    data_set = load_data()

    k = 8

    # Instantiate object for clustering
    model = KMeans(data_set, k)

    # Run the clustering algorithm
    model.fit()

    # Print the cluster points
    f = open(f"data/output_{k}_0.csv", "w")
    for i, cluster in enumerate(model.clusters):
        for c in cluster:
            f.write(f"{i},{c[0]},{c[1]}\n")
            print(c)
        print()
    f.close()

    # Print sizes of each cluster
    print("cluster_sizes: ", model.cluster_sizes)

    # Print the centroids
    print("centroids: ", model.centroids)
    f = open(f"data/centrois_{k}_0.csv", "w")
    for i, centroid in enumerate(model.centroids):
        f.write(f"{i},{centroid[0]},{centroid[1]}\n")
    f.close()


    # Print the Within Cluster Sum of Squares
    print("wcss: ", k, model.wcss)
