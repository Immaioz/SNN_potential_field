
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", type=int, required=True,
                        help="Numero di run della simulazione")
    parser.add_argument("--test", action="store_true", default=False,
                        help="Testing with online loop")
    parser.add_argument("--comparison", action="store_true", default=False,
                        help="Test comparison between online and offline")
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

    if args.comparison:
        online = True
        model = utils.load_model("SpikingAE_new.pth")
        np.random.seed(22) 
        seeds = np.random.randint(1, 1000, size=args.num_run)
        scene_path='C:/Users/User/Desktop/PField/potential_fields_sim_comparison.ttt'


    for i, run in enumerate(tqdm(range(args.num_run), desc="Simulation Runs")):
        run_path = os.path.join(save_path, f"Run_{run}_Seed_{seeds[i]}")
        os.makedirs(run_path, exist_ok=True)
        THR_base = 0.01087567
        simulator = Simulator(
            num_run=run,
            scene_path=scene_path if args.comparison else 'C:/Users/User/Desktop/PField/potential_fields_sim.ttt',
            # scene_path=scene_path if args.comparison else 'C:\\Users\\anton\\Documents\\PhD\\Spiking\\PotentialField_Sim\\potential_fields_sim.ttt',
            seed=seeds,
            num_blocks=20,
            min_distance=2.0,
            min_goal_distance=8.0,
            save_path=run_path,
            online=online,
            model=model,
            THR_base=THR_base,
            comparison=args.comparison
        )

        if args.comparison:
            results_dict, twin_results_dict = simulator.run()
        else:
            results_dict = simulator.run()
        
        utils.plot_trajectory(results_dict['pioneer_pos'], results_dict['block_pos'], results_dict['goal_pos'], results_dict['preds'], save=True, path = os.path.join(run_path, f"trajectory.png"))
        utils.plot_thr(results_dict['thresholds'], save=True, path = os.path.join(run_path, f"THR.png"))
        utils.plot_speed(results_dict['speed'], save=True, path = os.path.join(run_path, f"speed.png"))
        utils.plot_tot(results_dict['pioneer_pos'], results_dict['preds'], results_dict['thresholds'], results_dict['speed'], results_dict.get('arrival', None), save=True, path = os.path.join(run_path, f"resume.png"))

        np.savez_compressed(os.path.join(run_path, f"simulation_data.npz"),
                            pioneer_pos=results_dict['pioneer_pos'],
                            block_pos=results_dict['block_pos'],
                            goal_pos=results_dict['goal_pos'],
                            preds=results_dict['preds'],
                            thresholds=results_dict['thresholds'],
                            speeds=results_dict['speed'])
        
        np.savez_compressed(
        os.path.join(run_path, "simulation_data_dict.npz"),**results_dict)

        if args.comparison:
            utils.plot_trajectory(twin_results_dict['pioneer_pos'], twin_results_dict['block_pos'], twin_results_dict['goal_pos'], twin_results_dict['preds'], save=True, path = os.path.join(run_path, f"trajectory_twin.png"))
            utils.plot_thr(twin_results_dict['thresholds'], save=True, path = os.path.join(run_path, f"THR_twin.png"))
            utils.plot_speed(twin_results_dict['speed'], save=True, path = os.path.join(run_path, f"speed_twin.png"))
            utils.plot_tot(twin_results_dict['pioneer_pos'], twin_results_dict['preds'], twin_results_dict['thresholds'], twin_results_dict['speed'], twin_results_dict.get('arrival', None), save=True, path = os.path.join(run_path, f"resume_twin.png"))

            np.savez_compressed(os.path.join(run_path, f"simulation_data_twin.npz"),
                                pioneer_pos=twin_results_dict['pioneer_pos'],
                                block_pos=twin_results_dict['block_pos'],
                                goal_pos=twin_results_dict['goal_pos'],
                                preds=twin_results_dict['preds'],
                                thresholds=twin_results_dict['thresholds'],
                                speeds=twin_results_dict['speed'])
            
            np.savez_compressed(
            os.path.join(run_path, "simulation_data_dict_twin.npz"),**twin_results_dict)   

        
        anomalies = np.bincount(results_dict['preds'])[-1].item()
        normal = np.bincount(results_dict['preds'])[0].item()
        msg = f"Run {i} completata con {anomalies} anomalie su {normal+anomalies} totali"
        utils.send_telegram(msg)

if __name__ == "__main__":
    main()
