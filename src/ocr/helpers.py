import os
import re
import sys
import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def add_path_init():
    print("Add link config, datasets, src, tools to path.")
    current_directory = os.getcwd()
    directories = ["datasets", "config", "tools", "src"]
    for directory in directories:
        sys.path.insert(0, os.path.join(current_directory, directory))


def convert_contours_to_bounding_boxes(contours):
    """
    Convert a list of contours to a list of bounding boxes.

    Args:
    - contours (list): List of contours obtained from an image.

    Returns:
    - boxes (list): List of bounding boxes corresponding to the contours.
    """
    bounding_boxes = []

    # Iterate through each contour
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Define the bounding box using top-left and bottom-right coordinates
        box = (x, y, x + w, y + h)
        # Append the bounding box to the list
        bounding_boxes.append(box)

    # Sort the bounding boxes based on their x-coordinate
    bounding_boxes.sort(key=lambda box: box[0])

    return bounding_boxes


def group_boxes(boxes):
    boxes.sort(key=lambda box: box[0])
    # Convert the data to numpy array
    data_np = np.array(boxes)
    # Define the DBSCAN parameters
    epsilon = 50  # Distance threshold
    min_samples = 2  # Minimum number of samples in a cluster

    # Create the DBSCAN model
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)

    # Fit the model to the data
    dbscan.fit(data_np)

    # Get the labels assigned to each point
    labels = dbscan.labels_

    # Get the unique labels (clusters)
    unique_labels = np.unique(labels)

    cluster_boxes = {}

    # Calculate bounding boxes for each cluster
    for label in unique_labels:
        # if label == -1:
        #     print("Noise points:")
        if label != -1:
            # print("Cluster", label, ":")
            cluster_points = data_np[labels == label]
            min_x = np.min(cluster_points[:, 0])
            max_x = np.max(cluster_points[:, 2])
            min_y = np.min(cluster_points[:, 1])
            max_y = np.max(cluster_points[:, 3])
            cluster_boxes[label] = (min_x, min_y, max_x, max_y)

    return cluster_boxes
