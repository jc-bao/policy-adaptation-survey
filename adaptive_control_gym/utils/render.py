import bpy
from mathutils import Quaternion, Vector

# Set up the scene
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 100

# Set the file path to the STL file
stl_file_path = "/home/pcy/rl/policy-adaptation-survey/adaptive_control_gym/assets/crazyflie2.stl"

# Load the STL file
bpy.ops.import_mesh.stl(filepath=stl_file_path)

# Get a reference to the imported object
obj = bpy.context.selected_objects[0]

# Set up animation
obj.animation_data_create()
obj.animation_data.action = bpy.data.actions.new(name="Animation")

# Define the orientations and positions as quaternions and vectors
orientations = [
    Quaternion((1, 0, 0, 0)),  # Quaternion 1
    Quaternion((0, 1, 0, 0)),  # Quaternion 2
    Quaternion((0, 0, 1, 0))   # Quaternion 3
]

positions = [
    Vector((0, 0, 0)),    # Position 1
    Vector((1, 0, 0)),    # Position 2
    Vector((0, 1, 0))     # Position 3
]

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
