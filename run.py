
from simulator import Simulator
from utils import utils
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
import requests
import time
from config import BOT_TOKEN, CHAT_ID


def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg}
    requests.post(url, data=payload)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", type=int, required=True,
                        help="Numero di run della simulazione")
    parser.add_argument("--test", action="store_true", default=False,
                        help="Testing with online loop")
    args = parser.parse_args()

    save_path = './simulation_data_online' if args.test else './simulation_data'
    os.makedirs(save_path, exist_ok=True)
    if args.test:
        online = True
        model = utils.load_model("SpikingAE_new.pth")
        np.random.seed(22) 
        seeds = np.random.randint(1, 1000, size=args.num_run)
    else:
        online = False
        model = None
        seeds = [5, 26, 33, 58, 91, 73, 88, 12, 5, 39, 47, 61, 79, 84, 95, 14, 27, 42, 22, 17]

    for i, run in enumerate(tqdm(range(args.num_run), desc="Simulation Runs")):
        run_path = os.path.join(save_path, f"Run_{run}_Seed_{seeds[i]}")
        os.makedirs(run_path, exist_ok=True)
        THR_base = 0.01087567
        simulator = Simulator(
            num_run=run,
            scene_path='C:/Users/User/Desktop/PField/potential_fields_sim.ttt',
            # scene_path='C:\\Users\\anton\\Documents\\PhD\\Spiking\\PotentialField_Sim\\potential_fields_sim.ttt',
            seed=seeds,
            num_blocks=20,
            min_distance=2.0,
            min_goal_distance=8.0,
            save_path=run_path,
            online=online,
            model=model,
            THR_base=THR_base
        )

        pioneer_pos, block_pos, goal_pos, preds, thresholds, speed = simulator.run()

        utils.plot_trajectory(pioneer_pos[0], block_pos[0], goal_pos[0], preds[0], save=True, path = os.path.join(run_path, f"trajectory.png"))
        utils.plot_thr(thresholds[0], save=True, path = os.path.join(run_path, f"THR.png"))
        utils.plot_speed(speed[0], save=True, path = os.path.join(run_path, f"speed.png"))
        utils.plot_tot(pioneer_pos[0], preds[0], thresholds[0], speed[0], save=True, path = os.path.join(run_path, f"resume.png"))

        np.savez_compressed(os.path.join(run_path, f"simulation_data.npz"),
                            pioneer_pos=pioneer_pos[0],
                            block_pos=block_pos[0],
                            goal_pos=goal_pos[0],
                            preds=preds[0],
                            thresholds=thresholds[0],
                            speeds=speed[0])
        anomalies = np.bincount(preds[0])[-1].item()
        normal = np.bincount(preds[0])[0].item()
        msg = f"Run {i} completata con {anomalies} anomalie su {normal+anomalies} totali"
        send_telegram(msg)

if __name__ == "__main__":
    main()
