import torch
import yaml
import argparse 
import random
import numpy

from hed_pipeline import *
from attrdict import AttrDict

###############
# parse cfg
###############
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', dest='cfg', required=True, help='path to config file')
parser.add_argument('--mode', dest='mode', required=True, help='path to config file')
parser.add_argument('--time', dest='time', required=True, help='path to config file')
args = parser.parse_args()

#print(args)
cfg_file = args.cfg
print('cfg_file: ', cfg_file)
#print('mode: ', type(args.mode))

with open('config/'+cfg_file, 'r') as f:
    cfg = AttrDict( yaml.load(f) )

cfg.path = cfg_file
cfg.time = args.time
print(cfg)


random_seed = cfg.TRAIN.random_seed
if random_seed>0:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    numpy.random.seed(random_seed)


########################################

hed_pipeline = HEDPipeline(cfg)

if args.mode=='train':
    hed_pipeline.train()
    #import cProfile
    #cProfile.run( "hed_pipeline.train()", filename="a.out" )
else: 
    hed_pipeline.test()




