import laspy
import numpy as np

def subsample_point_cloud(input_file, factor=0.1):
    """
    Subsamples a point cloud from a LAS file by a given factor.

    Parameters:
    input_file (str): Path to the input LAS file.
    factor (int): Percentage of points to retain (default is 10).

    Returns:
    laspy.LasData: A new LasData object containing the subsampled points.
    """
    # Open the input LAS file
    with laspy.open(input_file) as las_file:
        # Read the entire point cloud
        las_data = las_file.read()
        
        # Get the points as a structured array
        points = las_data.points
            
        # Randomly subsample the points
        sample_size = int(len(points) * factor)
        random_indices = np.random.choice(len(points), sample_size, replace=False)
        reduced_points = points[random_indices]

        # Create a new LasData object with the same header as the original
        reduced_las = laspy.LasData(header=las_data.header, points=reduced_points)
        return reduced_las


if __name__ == '__main__':
    las = subsample_point_cloud('../../data/cloud_merged.las', factor=50)

    # Extract XYZ coordinates
    mean_x = np.mean(las.x)
    mean_y = np.mean(las.y)
    mean_z = np.mean(las.z)
    x = las.x - mean_x
    y = las.y - mean_y
    z = las.z - mean_z

    print(np.min(x), np.max(x))
    print(np.min(y), np.max(y))
    print(np.min(z), np.max(z))