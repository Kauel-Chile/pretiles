import laspy
import numpy as np

def subsample_point_cloud(input_file, output_file, factor=10):
    # Open the input LAS file
    with laspy.open(input_file) as las_file:
        # Read the entire point cloud
        las_data = las_file.read()
        
        # Get the points as a structured array
        points = las_data.points
            
        # Randomly subsample the points
        sample_size = int(len(points) * (factor / 100))
        random_indices = np.random.choice(len(points), sample_size, replace=False)
        reduced_points = points[random_indices]

        # Create a new LasData object with the same header as the original
        reduced_las = laspy.LasData(header=las_data.header, points=reduced_points)
        
        # Write the reduced point cloud to a new LAS file
        reduced_las.write(output_file)