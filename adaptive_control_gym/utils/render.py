import bpy
from mathutils import Quaternion, Vector
import csv

# Set up the scene
scene = bpy.context.scene
scene.frame_start = 1
scene.render.fps = int(1/4e-4)

# Set the file path to the STL file
stl_file_path = "/home/pcy/rl/policy-adaptation-survey/adaptive_control_gym/assets/crazyflie2.stl"

# Load the STL file
bpy.ops.import_mesh.stl(filepath=stl_file_path)

# Get a reference to the imported object
obj = bpy.context.selected_objects[0]

# Set up animation
obj.animation_data_create()
obj.animation_data.action = bpy.data.actions.new(name="Animation")

# Load csv file as a dict
orientations = [ ]
positions = [ ]
csv_file_path = "/home/pcy/rl/policy-adaptation-survey/adaptive_control_gym/envs/results/test.csv"
num_frame = 0
with open(csv_file_path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        orientation = Quaternion((float(row["quat_drones_3"]), float(row["quat_drones_0"]), float(row["quat_drones_1"]), float(row["quat_drones_2"])))
        position = Vector((float(row["xyz_drones_0"]), float(row["xyz_drones_1"]), float(row["xyz_drones_2"])))
        orientations.append(orientation)
        positions.append(position)
        num_frame += 1
scene.frame_end = num_frame

# Create keyframes for each frame in the animation
for frame in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(frame)
    
    # Calculate the index for the current frame
    index = (frame - 1) % len(orientations)
    
    # Set the object's orientation and position for the current frame
    obj.rotation_quaternion = orientations[index]
    obj.location = positions[index]
    
    # Insert keyframes for rotation and position
    obj.keyframe_insert(data_path="rotation_quaternion")
    obj.keyframe_insert(data_path="location")

# Save the Blender project file
save_file_path = "/home/pcy/rl/policy-adaptation-survey/adaptive_control_gym/utils/results/project.blend"
bpy.ops.wm.save_as_mainfile(filepath=save_file_path)