# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter
import numpy as np
from math import isclose
import bmesh
import statistics
from mathutils import Matrix, Vector, Quaternion

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.") 
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=10, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--rot_angle', default=20, type=float, 
    help="Amount of rotation for objects between placing consecutive leaves")
parser.add_argument('--max_retries', default=50, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_scenes', default=5, type=int,
    help="The number of scenes to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='../output/images/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='output/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--output_obj_dir', default='output/objfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_objfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--save_objfiles', type=int, default=1,
    help="Setting --save_objfiles 0 will cause blender scene file for each " +
        "image to NOT be stored in the directory. These files are saved by " +
        "default for use in neural net.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=256, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=256, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")

def main(args):
  num_digits = 6
  prefix = '{}_{}'.format(args.filename_prefix, args.split)
  img_template = '%%0%dd.png' % (num_digits)  # file name 
  img_id = '%s%%0%dd' % (prefix, num_digits)  # folder name
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  obj_template = '%s%%0%dd' % (prefix, num_digits)  # folder name for obj files for each scene
  # single_img_template = os.path.join(args.output_image_dir, img_id, img_template)  # template for filepath for each scene
  single_img_template = img_template
  print("single img: ", single_img_template, "& obj template: ", obj_template)
  img_template = os.path.join(args.output_image_dir, img_id)   # template for folderpath for scenes
  scene_template = os.path.join(args.output_scene_dir, scene_template)
  blend_template = os.path.join(args.output_blend_dir, blend_template)
  obj_template = os.path.join(args.output_obj_dir, obj_template)


  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)
  if args.save_objfiles == 1 and not os.path.isdir(args.output_obj_dir):
    os.makedirs(args.output_obj_dir)
  
  all_scene_paths = []
  vol_list = []
  num_images_per_scene = 50

  for i in range(args.num_scenes):
    # for j in range(num_images_per_scene):
    img_path = single_img_template  # only the filename. Needs os.path.join in the render_scene function
    print("img path: ", img_path)
    folder_path = img_template  % (i + args.start_idx)
    scene_path = scene_template % (i + args.start_idx)
    all_scene_paths.append(scene_path)
    blend_path = None
    obj_path = None
    if args.save_blendfiles == 1:
      blend_path = blend_template % (i + args.start_idx)
    if args.save_objfiles == 1:
      obj_path = obj_template % (i + args.start_idx)
    print("obj path: ", obj_path)
    num_objects = random.randint(args.min_objects, args.max_objects)
    # Get volume of objs in each scene    
    vol_objs = render_scene(args,
      num_objects=num_objects,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=img_path,   # img path being given here is the single image template. The format int placement will be in function itself
      output_folder=folder_path,
      output_scene=scene_path,
      output_blendfile=blend_path,
      output_obj_folder=obj_path,
      num_imgs_per_scene=num_images_per_scene
    )
    vol_list.append(vol_objs)

  # After rendering all images, combine the JSON files for each scene into a
  # single JSON file.
  all_scenes = []
  for scene_path in all_scene_paths:
    with open(scene_path, 'r') as f:
      all_scenes.append(json.load(f))
  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': all_scenes
  }
  with open(args.output_scene_file, 'w') as f:
    json.dump(output, f)

  print("Volume of scenes in dataset: ", vol_list)
  if len(vol_list) > 2:
    std_dev = get_std_deviation_volume(vol_list) # std deviation across dataset
    print("Std deviation: ", std_dev)

def _register_meshes():
  # Register all mesh objects together to group later
  meshes = []
  for obj in bpy.data.objects:
    if obj.type == "MESH":
      meshes.append(obj)
  return meshes

def config_cam(azimuth, elevation, dist, tilt=0):
    """
    Author: Amit Raj
    Utility function to generate camera location and rotation Vectors    
    Parameters:
      azimuth: float
          Azimuth angle in radians
      elevation: float
          Elevation angle in radians
      dist : float
          Distance of viewing sphere
      tilt : float
          In Plane rotation in radians    
    Returns:
      location : mathutils.Vector
          Camera location setting
      rotation : mathutils.Vector
          Camera rotation setting
    """    
    z = dist * np.sin(elevation)
    y = -dist * np.cos(azimuth) * np.cos(elevation)
    x = dist * np.sin(azimuth) * np.cos(elevation)    
    location = Vector((x, y, z))   
    xr = np.pi / 2 - elevation
    yr = tilt
    zr = azimuth    
    rotation = Vector((xr, yr, zr))    
    return location, rotation


def convert_to_array(matrix_obj):
  ''' Input: 3x3 matrix from mathutils
  Output: Array type, for bpy rotate function
  '''
  array = np.array(np.vstack(([matrix_obj[i] for i in range(len(matrix_obj))])))
  print(array)
  return array


def get_mesh_from_obj(ob):
  matrix = ob.matrix_world
  ob_mesh = ob.to_mesh(bpy.context.scene, apply_modifiers=True, settings='PREVIEW')
  ob_mesh.transform(matrix)
  # ob_mesh = ob.data
  bm = bmesh.new()
  bm.from_mesh(ob_mesh)
  bpy.data.meshes.remove(ob_mesh)
  # TEST CASE: for testing alternative approach
  # is_neg = False
  return bm, matrix.is_negative

def get_std_deviation_volume(vol):
  # Standard deviation of volumes across dataset
  # Input: vol(list)- list of scene volumes of all images in dataset
  std_deviation = statistics.stdev(vol)
  return std_deviation

'''
Rotation function; Reference: https://devtalk.blender.org/t/bpy-ops-transform-rotate-option-axis/6235
'''
def create_z_orient(rot_vec):
    x_dir_p = Vector(( 1.0,  0.0,  0.0))
    y_dir_p = Vector(( 0.0,  1.0,  0.0))
    z_dir_p = Vector(( 0.0,  0.0,  1.0))
    tol = 0.001
    rx, ry, rz = rot_vec
    if isclose(rx, 0.0, abs_tol=tol) and isclose(ry, 0.0, abs_tol=tol):
        if isclose(rz, 0.0, abs_tol=tol) or isclose(rz, 1.0, abs_tol=tol):
            return Matrix((x_dir_p, y_dir_p, z_dir_p))  # 3x3 identity
    new_z = rot_vec.copy()  # rot_vec already normalized
    new_y = new_z.cross(z_dir_p)
    new_y_eq_0_0_0 = True
    for v in new_y:
        if not isclose(v, 0.0, abs_tol=tol):
            new_y_eq_0_0_0 = False
            break
    if new_y_eq_0_0_0:
        new_y = y_dir_p
    new_x = new_y.cross(new_z)
    new_x.normalize()
    new_y.normalize()
    return Matrix(((new_x.x, new_y.x, new_z.x),
                   (new_x.y, new_y.y, new_z.y),
                   (new_x.z, new_y.z, new_z.z)))

def render_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_folder='render',
    output_scene='render_json',
    output_blendfile=None,
    output_obj_folder=None,
    num_imgs_per_scene=60
  ):
  
  if not os.path.isdir(output_obj_folder):
    os.makedirs(output_obj_folder)
  
  # TEST CASE: PLace one object only
  # num_objects = 1

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
  ground_plane = bpy.context.object

  # Load materials
  utils.load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100
  render_args.tile_x = args.render_tile_size
  render_args.tile_y = args.render_tile_size
  if args.use_gpu == 1:
    # Blender changed the API for enabling CUDA at some point
    if bpy.app.version < (2, 78, 0):
      bpy.context.user_preferences.system.compute_device_type = 'CUDA'
      bpy.context.user_preferences.system.compute_device = 'CUDA_0'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific stuff
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'image_folder': os.path.basename(output_folder),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)
  
  # Get camera location and rotation to move in a 'viewing sphere' per scene
  camera = bpy.data.objects["Camera"]            # Sample random Azimuth and elevation

 
  # # Add random jitter to camera position
  # if args.camera_jitter > 0:
  #   for i in range(3):
  #     bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  # camera = bpy.data.objects['Camera']
  plane_normal = plane.data.vertices[0].normal
  cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # Delete the plane; we only used it for normals anyway. The base scene file
  # contains the actual ground plane.
  utils.delete_object(plane)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera)
  # Objects is a dict of scene obj properties(shape, color, etc) and blender_objects is a list of mesh objects in scene

  # We want objects in .obj format, for PIFu: for this, we group rendered objects 
  # together and save geometry to .obj. However, this also groups the ground plane
  # with the objects, which is unnecessary.

  # Delete ground plane, to make grouping for .obj easier
  utils.delete_object(ground_plane)
  
  vol, vol_obj = 0.0, 0.0
  # Group together meshes
  bpy.ops.group.create(name="meshGroup")
  mesh_list = _register_meshes()
  for ob in mesh_list:
    bpy.context.scene.objects[ob.name].select = True
    bpy.ops.object.group_link(group="meshGroup") 
    bm, is_negative = get_mesh_from_obj(ob)
    bpy.context.scene.objects[ob.name].select = False  
  
    
    vol_obj = bm.calc_volume(signed=True)  # Get volume for a single object
    print("debugging vol func: ", vol_obj, " for is_neg flag: ", is_negative)
    if is_negative:
        vol_obj = -vol_obj
    bm.free()
    vol += vol_obj
  
  print("Volume of scene: ", vol)
  # At the end of this, the central sphere is the active object, as seen by bpy.context.object
  # Camera viewing angles
  target_obj = bpy.context.object
  target_loc_x, target_loc_y = target_obj.location.x, target_obj.location.y
  cam_loc_x, cam_loc_y = camera.location.x, camera.location.y
  base_dist = (target_obj.location.xy-camera.location.xy).length

  # TRIAL RUN:
  # ISSUES: All files getting saved in one folder
  # viewing sphere seems to be around z-axis, instead of whatever we want
  # files getting saved on top of each other - 3 views of same scene overwriting each other
  t_list = np.linspace(0,1,num_imgs_per_scene)
  for j in range(num_imgs_per_scene):
    img = os.path.join(output_folder, output_image) % j
    render_args.filepath = img
    print("render args filepath: ", output_folder, " joined: ", img)
    t = t_list[j]
    # t = 0.5
    azimuth = 180 + (-180 * t + (1 - t) * 180)  # range of azimuth: 0-360 deg
    elevation = 15 * t + (1 - t) * 75        # range of elevation: 15-75 deg    
    jitter_t = np.random.rand()   # Jitter the viewing sphere
    jitter = -args.camera_jitter * (1 - jitter_t) + jitter_t * args.camera_jitter
    dist = base_dist + jitter            
    location, rotation = config_cam(
        math.radians(azimuth), math.radians(elevation), dist
    )            
    camera.location = location
    camera.rotation_euler = rotation
      # Render the scene and dump the scene data structure
    scene_struct['objects'] = objects
    scene_struct['relationships'] = compute_all_relationships(scene_struct)
    while True:
      try:
        bpy.ops.render.render(write_still=True)
        break
      except Exception as e:
        print(e)

    obj_file_template = '%%0%dd.obj' % 6  # num_digits hardcoded here
    print(" export scene obj path: ", os.path.join(output_obj_folder, obj_file_template) % j)
    bpy.ops.export_scene.obj(filepath=os.path.join(output_obj_folder, obj_file_template) % j)

    with open(output_scene, 'w') as f:
      json.dump(scene_struct, f, indent=2)

    bpy.ops.object.select_all(action='DESELECT')

    if output_blendfile is not None:
      bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
  
  return vol

def add_random_objects(scene_struct, num_objects, args, camera):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    

    # Choose between disks.blend and leaf.obj
    leaf_disk = False
    leaf_properties = properties['shapes'].copy()
    leaf_properties.pop('sphere')
    leaf_obj_mapping = [(v,k) for k,v in leaf_properties.items()]
    size_mapping = list(properties['sizes'].items())
    print("size mapping: ", size_mapping)

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []

  init_rot = 0
  size_name, r = random.choice(size_mapping)
  color_name, rgba = random.choice(list(color_name_to_rgba.items()))

  '''
  WORKFLOW: First select nb of rows - 1 being bottom-most
  For each row, select nb of leaves - each row has a tilt angle that's a function of the row
  For each leaf, select angular position of leaf around z-axis. ensure leaves don't fully overlap
  '''
  nb_rows = random.randint(1,3)
  
  # TEST CASE: 
  # nb_rows = 4
   
  size_map_sorted = sorted(size_mapping, key = lambda x:x[1])
  size_map_sorted.reverse()

  # Position of leaves
  x = 0.0
  y = 0.0

  for row in range(nb_rows):
    # For each row pick nb of leaves
    nb_leaves_per_row = random.randint(3,7) 

    # TEST CASE:
    #nb_leaves_per_row = 4  # nb_leaves + 1 (coz sphere) --> change

    for i in range(nb_leaves_per_row):

      theta = math.pi*(random.choice(list(np.arange(row*20, (row+1)*20, 0.5))))/180 # lower leaves are less tilted

      # Pick largest leaves for lowest rows, and decrease leaf size as rows increase (may need to be modified for more randomness)
      #size_name, r = size_map_sorted[row]
      size_name, r = random.choice(size_mapping)
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
      if not positions:
        z = r
        obj_name_out = str('sphere')
        obj_name = properties['shapes']['sphere']
      else:
        z = 0.0        
        if leaf_disk == True:
          obj_name, obj_name_out = random.choice(leaf_obj_mapping)
        else:
          obj_name_out = str('leaf')
          obj_name = str('leaf')
      # Actually add the object to the scene
      utils.add_object(args.shape_dir, obj_name, r, (x, y,z), theta=theta)
      obj = bpy.context.object
      blender_objects.append(obj)
      positions.append((x, y, r))

      # TEST CASE: CHECK ORIENTATION
      print("theta: ", theta, " for obj: ", obj)

      # Attach a random material
      mat_name, mat_name_out = random.choice(material_mapping)
      utils.add_material(mat_name, Color=rgba)
      # Attach a random material
      mat_name, mat_name_out = random.choice(material_mapping)
      utils.add_material(mat_name, Color=rgba)

      # Record data about the object in the scene data structure
      pixel_coords = utils.get_camera_coords(camera, obj.location)
      objects.append({
        'shape': obj_name_out,
        'size': size_name,
        'material': mat_name_out,
        '3d_coords': tuple(obj.location),
        'rotation': theta,
        'pixel_coords': pixel_coords,   
        'color': color_name,
      })

      # Rotate objects on axis of central sphere before placing next object
      # Create a parent and add children
      parent_object = blender_objects[0]
      bpy.context.object.rotation_mode = 'XYZ'
      
      if (len(positions) > 1 and obj.type == "MESH"):
        obj.select = True
        parent_object.select = True
        obj.parent = parent_object
        obj.matrix_parent_inverse = parent_object.matrix_world.inverted()

      rot_angle_deg = 360.0 / (float(nb_leaves_per_row-1)) # to have some overlap between objects
      rot_angle = (3.14159 * rot_angle_deg / 180 )  
      # print("Rotated by: ", rot_angle - init_rot, " & rotation angle was: ", rot_angle)
      init_rot = rot_angle
    
      # Info note: The axis of interest is indexed by bpy.data.objects['ObjName'].rotation_euler[2]
      bpy.context.scene.objects.active = bpy.data.objects[parent_object.name]
      bpy.context.object.rotation_euler[2] = bpy.context.object.rotation_euler[2] + rot_angle
      parent_object.select = False  
      obj.select = False
      bpy.ops.object.select_all(action='DESELECT')

  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)

  return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.

  Returns True if all objects are visible and False otherwise.
  """
  f, path = tempfile.mkstemp(suffix='.png')
  object_colors = render_shadeless(blender_objects, path=path)
  img = bpy.data.images.load(path)
  p = list(img.pixels)
  color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                        for i in range(0, len(p), 4))
  os.remove(path)
  if len(color_count) != len(blender_objects) + 1:
    return False
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False
  return True


def render_shadeless(blender_objects, path='flat.png'):
  """
  Render a version of the scene with shading disabled and unique materials
  assigned to all objects, and return a set of all colors that should be in the
  rendered image. The image itself is written to path. This is used to ensure
  that all objects will be visible in the final rendered scene.
  """
  render_args = bpy.context.scene.render

  # Cache the render args we are about to clobber
  old_filepath = render_args.filepath
  old_engine = render_args.engine
  old_use_antialiasing = render_args.use_antialiasing

  # Override some render settings to have flat shading
  render_args.filepath = path
  render_args.engine = 'BLENDER_RENDER'
  render_args.use_antialiasing = False

  # Move the lights and ground to layer 2 so they don't render
  utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
  utils.set_layer(bpy.data.objects['Ground'], 2)

  # Add random shadeless materials to all objects
  object_colors = set()
  old_materials = []
  for i, obj in enumerate(blender_objects):
    old_materials.append(obj.data.materials[0])
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % i
    while True:
      r, g, b = [random.random() for _ in range(3)]
      if (r, g, b) not in object_colors: break
    object_colors.add((r, g, b))
    mat.diffuse_color = [r, g, b]
    mat.use_shadeless = True
    obj.data.materials[0] = mat

  # Render the scene
  bpy.ops.render.render(write_still=True)

  # Undo the above; first restore the materials to objects
  for mat, obj in zip(old_materials, blender_objects):
    obj.data.materials[0] = mat

  # Move the lights and ground back to layer 0
  utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
  utils.set_layer(bpy.data.objects['Ground'], 0)

  # Set the render settings back to what they were
  render_args.filepath = old_filepath
  render_args.engine = old_engine
  render_args.use_antialiasing = old_use_antialiasing

  return object_colors


if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')

