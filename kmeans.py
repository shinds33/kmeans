import pandas as pd
import numpy as np
import sys
import random

# Gathering user data and setting up IO
stdoutOrigin = sys.stdout
sys.stdout = open("output.txt", "w")
current_input = sys.argv[2]
in_data = pd.read_csv(current_input, sep="\t", header=None)
unknowns = np.array(in_data)


class k_means:
    k_value = sys.argv[1]

    def __init__(self, k=int(k_value), tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}
        input_size = data.size

        # Starting centroids chosen randomly
        for i in range(self.k):
            current_cent = random.randint(0, ((input_size/2))-1)
            self.centroids[i] = data[current_cent]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureSet in data:
                distances = [np.linalg.norm(featureSet - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureSet)

            previous_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = previous_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = k_means()
clf.fit(unknowns)

for unknown in unknowns:
    classification = clf.predict(unknown)
    print(unknown, end=" ")
    print(classification+1)

sys.stdout.close()
sys.stdout = stdoutOrigin



