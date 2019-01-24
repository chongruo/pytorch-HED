import torch
import yaml
import argparse 

from dataset.BSD500 import BSD500Dataset
from models.HED import HED


###############
# parse cfg
###############
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', dest='cfg', required=True, help='path to config file')
args = parser.parse_known_args()
args = parser.parse_args()

#print(args)
cfg_file = args.cfg
print('cfg_file: ', cfg_file)

with open('config/'+cfg_file, 'r') as f:
    cfg = yaml.load(f)

print(cfg)


########################################


model = HED(cfg)































