#!/usr/bin/env python

import os
import subprocess
import sys
import glob
import ntpath

ROOT_DIR = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(ROOT_DIR)
print os.path.join(ROOT_DIR, 'heatmap.py')
files = glob.glob('data/dia/*.shp')
for file in files:
    output_file = 'data/output_tiffs/'+ ntpath.basename(file).split(".",1)[0] + '.png'
    subprocess.check_call(
        ['./heatmap.py',
         '--shp_file', file,
         '-r', '15',
         '-W', '1000',
         '--osm',
         '-B','0.5',
         '--osm_base',' http://b.tile.stamen.com/toner',
         '--extent', '18.9584,-99.542,19.9373,-98.4822',
         '-o', output_file])
