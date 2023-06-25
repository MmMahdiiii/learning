#! /usr/bin/env python
# -*- coding: utf-8 -*-

# python imports
import os
import sys

# chillin imports
from chillin_client import GameClient

# project imports
from ai import AI
from ks.models import World

# json and pickle
import json
import pickle


config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs\\gamecfg0.json"
)
if len(sys.argv) > 1:
    config_path = sys.argv[1]


config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs\\gamecfg0.json"
)
if len(sys.argv) > 1:
    config_path = sys.argv[1]

print('Loading config from {}'.format(config_path))
with open(config_path, 'r') as f:
    conf = json.load(f)
    nn_path = conf['ai']['nn_path']

print('Loading neural network from {}'.format(nn_path))
with open(nn_path, 'rb') as nn_file:
    nn = pickle.load(nn_file)

print('Starting client')
ai = AI(World(), nn)

app = GameClient(config_path)
app.register_ai(ai)
app.run()
