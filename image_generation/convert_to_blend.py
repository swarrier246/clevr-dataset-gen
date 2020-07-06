# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import sys, random, os
import bpy, bpy_extras


"""
Some utility functions for interacting with Blender
"""
file_path = "data/shapes"

for fn in os.listdir(file_path):
  if fn.endswith('.obj'):
    name = fn.split('.')[0]
    filename = os.path.join(file_path, '%s.obj' % name)
    output_filename = os.path.join(file_path, '%s.blend' % name)
    bpy.ops.import_scene.obj(filepath=filename)
    bpy.ops.wm.save_as_mainfile(filepath=output_filename)