'''
This script combines the ground states code written by Ellie Copps, number_components written by Jack Landrigan and
the perturbation code written by Tao You to perform comprehensive analysis of quantum spin glasses

Work flow:

import main() NN_ground.py
    --input denoted parameters,
    --save the data to a txt file
    --read the txt file, it is going to be a python dictionary
    --then attach the analysis code, for any seed instance just extract from the python dictionary

import number_components(N, ground_state) where ground_states is a list of indices, exactly the output of Ellie's code
    --input denoted parameters
    --returns dist_comp, which is a dictionary {hamming_distance: number of connected components}, the correct perturbative order
'''

from tfim_survey import tfim_analysis
import os
os.chdir('../NN_Ground')
print(os.getcwd())
from NN_ground import main
from connected_components import number_components
os.chdir('../tfim_perturbation')
print(os.getcwd())