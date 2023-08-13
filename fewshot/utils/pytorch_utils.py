import json

import os

import torch


def save(model, niter, save_folder):
  if not os.path.exists(save_folder):
    os.makedirs(save_folder)
  torch.save(model, os.path.join(save_folder, "model" + str(niter) + ".pt"))

def save_config(config, save_folder):
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
  config_file = os.path.join(save_folder, "conf.json")
  with open(config_file, "w") as f:
    f.write(json.dumps(dict(config.__dict__)))

def save_config(config, save_folder):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    config_file = os.path.join(save_folder, "conf-pytorch.json")
    with open(config_file, "w") as f:
        f.write(json.dumps(dict(config.__dict__)))

def isnan(x):
    return x != x
