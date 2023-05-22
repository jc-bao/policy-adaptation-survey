import bpy
import mathutils

def set_keyframe(obj, data_path, frame, value):
    obj.keyframe_insert(data_path=data_path, frame=frame)
    obj.animation_data.action.fcurves[-1].keyframe_points[-1].interpolation = 'LINEAR'
    obj.animation_data.action.fcurves[-1].keyframe_points[-1].co = frame, value

def create_animation(stl_file, orientations, positions):
    # delete scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # import file
    bpy.ops.import_mesh.stl(filepath=stl_file)
    mesh = bpy.context.object

    # add plane light which orientates to center of the world
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 5.0))
    bpy.context.object.data.energy = 1
    # add camera whose distance to target is 1m
    cam_data = bpy.data.cameras.new('camera')
    cam = bpy.data.objects.new('camera', cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.location = (0, 0.5, 0.5)
    cam.rotation_euler = (mathutils.Euler((0, 0, 0), 'XYZ'))
    cam.data.lens = 35
    cam.data.clip_start = 0.1
    cam.data.clip_end = 1000
    # add blue ground plane to make the object more visible
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -1.0))
    
    # setup render parameters
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.audio_codec = 'AAC'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.fps = 24
    # scene.render.engine = "CYCLES"
    # # Set the device_type
    # bpy.context.preferences.addons[
    #     "cycles"
    # ].preferences.compute_device_type = "CUDA"  # or "OPENCL" for non-NVIDIA GPUs
    # # Set the device and feature set
    # scene.cycles.device = "GPU"
    # bpy.context.preferences.addons["cycles"].preferences.get_devices()
    # print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    # for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    #     d["use"] = 1  # Using all devices, including GPU and CPU
    #     print(d["name"], d["use"])
    # scene = bpy.context.scene
    # scene.render.engine = 'BLENDER_EEVEE'
    # scene.render.eevee.use_gpu = True
    # scene.render.eevee.taa_render_samples = 1
    # scene.render.eevee.taa_samples = 1
    # scene.render.eevee.motion_blur_samples = 1
    # scene.render.eevee.use_motion_blur = False
    # scene.render.eevee.use_ssr = False
    # scene.render.eevee.use_bloom = False
    # scene.render.eevee.use_volumetric_shadows = False
    # scene.render.eevee.use_soft_shadows = False
    # scene.render.eevee.gi_diffuse_bounces = 0
    # scene.render.eevee.gi_cubemap_resolution = '512'
    # scene.render.eevee.taa_render_samples = 1
    # scene.render.eevee.taa_samples = 1
    # # Set the device to use for rendering
    # scene.cycles.device = 'GPU'
    # prefs = bpy.context.preferences
    # cprefs = prefs.addons['cycles'].preferences
    # cprefs.get_devices()
    # # Use all available GPUs
    # cprefs.compute_device_type = 'CUDA'
    # cprefs.devices[0].use = True

    # render
    scene.frame_start = 0
    scene.frame_end = len(orientations) - 1
    for frame, (orientation, position) in enumerate(zip(orientations, positions)):
        scene.frame_set(frame)
        bpy.context.active_object.rotation_quaternion = orientation
        bpy.context.active_object.location = position

    scene.render.filepath = 'results/blender_animation.mp4'
    bpy.ops.render.render(animation=True)

def main():
    orientations = [[0,0,0,1]]*2
    positions = [[0,0,i/10] for i in range(2)]
    create_animation("/home/pcy/rl/policy-adaptation-survey/adaptive_control_gym/assets/crazyflie2.stl", orientations, positions)

if __name__ == '__main__':
    main()