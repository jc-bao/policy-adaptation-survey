import bpy
import pandas as pd

# Path to the CSV file
csv_file = "../envs/results/test.csv"

# Path to the STL file
stl_file = "../assets/crazyfle2.stl"

# Read the CSV file using pandas
df = pd.read_csv(csv_file)

# Get the number of frames in the animation
num_frames = len(df)

# Set the scene frame range
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = num_frames

# Import the STL file as a mesh object
bpy.ops.import_mesh.stl(filepath=stl_file)
rigid_body = bpy.context.object

# Iterate over the rows in the DataFrame
for index, row in df.iterrows():
    # Get the position and rotation values from the current row
    position = (row['x'], row['y'], row['z'])
    rotation = (row['w'], row['x'], row['y'], row['z'])

    # Set the rigid body position and rotation for the current frame
    bpy.context.scene.frame_set(index + 1)
    rigid_body.location = position
    rigid_body.rotation_mode = 'QUATERNION'
    rigid_body.rotation_quaternion = rotation

    # Insert a keyframe for the rigid body's location and rotation
    rigid_body.keyframe_insert(data_path='location')
    rigid_body.keyframe_insert(data_path='rotation_quaternion')
