
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", type=int, required=True,
                        help="Numero di run della simulazione")
    args = parser.parse_args()

    save_path = './simulation_data'
    os.makedirs(save_path, exist_ok=True)

    seeds = [5, 26, 33, 58, 91, 73, 88, 12, 5, 39, 47, 61, 79, 84, 95, 14, 27, 42, 22, 17]

    simulator = Simulator(
        num_run=args.num_run,
        scene_path='C:/Users/User/Desktop/PField/potential_fields_sim.ttt',
        seed=seeds,
        num_blocks=20,
        min_distance=2.0,
        min_goal_distance=8.0,
        save_path=save_path
    )

    simulator.run()

if __name__ == "__main__":
    main()
