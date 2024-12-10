import numpy as np
from scipy.spatial import KDTree

def icp(source_points, target_points, max_iterations=100, tolerance=1e-6):
    """
    Perform Iterative Closest Point (ICP) algorithm.

    Parameters:
        source_points (np.ndarray): Source point cloud (Nx2 or Nx3).
        target_points (np.ndarray): Target point cloud (Nx2 or Nx3).
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.

    Returns:
        transformed_points (np.ndarray): Transformed source points aligned to target points.
        transformation (dict): Rotation and translation applied during alignment.
    """
    src = source_points.copy()
    tgt = target_points.copy()
    tree = KDTree(tgt)
    prev_error = float('inf')
    transformation = {"rotation": np.eye(src.shape[1]), "translation": np.zeros(src.shape[1])}

    for iteration in range(max_iterations):
        # Step 1: Find the closest points
        distances, indices = tree.query(src)
        closest_points = tgt[indices]

        # Step 2: Compute centroids
        src_centroid = np.mean(src, axis=0)
        tgt_centroid = np.mean(closest_points, axis=0)

        # Step 3: Center the points
        src_centered = src - src_centroid
        tgt_centered = closest_points - tgt_centroid

        # Step 4: Compute the rotation matrix using SVD
        H = np.dot(src_centered.T, tgt_centered)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # Ensure a proper rotation matrix (det(R) == +1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Step 5: Compute the translation vector
        T = tgt_centroid - np.dot(src_centroid, R)

        # Step 6: Apply the transformation
        src = np.dot(src, R) + T

        # Accumulate transformations
        transformation["rotation"] = np.dot(transformation["rotation"], R)
        transformation["translation"] = transformation["translation"] + T  # Accumulate translation correctly

        # Step 7: Check for convergence
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            print(f"Converged at iteration {iteration}")
            break
        prev_error = mean_error

    # Round final transformation for consistency
    transformation["rotation"] = np.round(transformation["rotation"], decimals=10)
    transformation["translation"] = np.round(transformation["translation"], decimals=10)

    return src, transformation
