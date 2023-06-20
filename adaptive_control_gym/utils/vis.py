import meshcat
from  meshcat import geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation, convert_frames_to_video
import numpy as np
from jax import numpy as jnp
import os
import time
import pickle

# Create a Meshcat visualizer
vis = meshcat.Visualizer()
anim = Animation(default_framerate=50)

# Add a box to the scene
box = g.Box([1, 1, 1])
vis["box"].set_object(box)

# load state sequence from pickle and check if load is successful
file_path = "../envs/results/state_seq.pkl"
with open(file_path, "rb") as f:
    state_seq = pickle.load(f)

# Apply the transformations according to the time sequence
for i, state in enumerate(state_seq):
    position = state.pos
    orientation = state.quat
    transform = tf.translation_matrix(position) @ tf.quaternion_matrix(orientation)
    with anim.at_frame(vis, i) as frame:
        frame["box"].set_transform(transform)
vis.set_animation(anim)
time.sleep(5)