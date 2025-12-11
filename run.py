
from simulator import Simulator
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import csv
import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import argparse
import os

from utils import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", type=int, required=True,
                        help="Numero di run della simulazione")
    parser.add_argument("--test", type=bool, action="store_true", default=False,
                        help="Testing with online loop")
    args = parser.parse_args()

    save_path = './simulation_data_online' if args.test else './simulation_data'
    os.makedirs(save_path, exist_ok=True)

    seeds = [5, 26, 33, 58, 91, 73, 88, 12, 5, 39, 47, 61, 79, 84, 95, 14, 27, 42, 22, 17]
    if args.test:
        online = True
        model = utils.load_model("SpikingAE_new.pth")
    else:
        online = False
        model = None

    THR_base = 0.01087567
    simulator = Simulator(
        num_run=args.num_run,
        # scene_path='C:/Users/User/Desktop/PField/potential_fields_sim.ttt',
        scene_path='C:\\Users\\anton\\Documents\\PhD\\Spiking\\PotentialField_Sim\\potential_fields_sim.ttt',
        seed=seeds,
        num_blocks=20,
        min_distance=2.0,
        min_goal_distance=8.0,
        save_path=save_path,
        online=online,
        model=model,
        THR_base=THR_base
    )

    pioneer_pos, block_pos, goal_pos, preds, thresholds = simulator.run()

    np.savez_compressed(os.path.join(save_path, "simulation_data.npz"),
                        pioneer_pos=pioneer_pos,
                        block_pos=block_pos,
                        goal_pos=goal_pos,
                        preds=preds,
                        thresholds=thresholds)

if __name__ == "__main__":
    main()
