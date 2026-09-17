import os
import shutil
import numpy as np
from tqdm import tqdm
import argparse
from simulator import Simulator, PotentialField, AdaptiveAPFPotentialField, AdaptiveTangentialAPFPotentialField
from utils import utils

PF_REGISTRY = {
    'PotentialField': PotentialField,
    'AdaptiveAPF': AdaptiveAPFPotentialField,
    'AdaptiveTangential': AdaptiveTangentialAPFPotentialField,
}


def setup_config(args):
    config = {
        'test': args.test,
        'online': False,
        'model': None,
        'model_class': None,
        'seed': 4,
        'scene_path': 'D:/Antonino/PField/potential_fields_sim.ttt',
        'save_path': './simulation_data_train_new',
        'THR_base': 0.02,
        'THR_base_class': 0.06581877,
        'decay_factor': 0.99,
    }
    if args.test:
        config.update({
            'online': True,
            'model': utils.load_model("REV_SpikingAE_opt.pth", num_inputs=800, num_outputs=800, num_hidden=384),
            'seed': 22,
            'scene_path': '/home/nino/PhD/Spiking/PotentialField_Sim/scenes/REV_potential_fields_sim.ttt',
            'save_path': 'simulation_data/REV_online_test_1',
        })
    if args.comparison:
        config.update({
            'online': True,
            'model': utils.load_model("SpikingAE_opt_1.pth", num_inputs=800, num_outputs=800, num_hidden=384),
            'seed': 17,
            'scene_path': '/home/nino/PhD/Spiking/PotentialField_Sim/scenes/REV_potential_fields_sim_comparison.ttt',
            'save_path': './REV_simulation_data_opt',
        })
    if args.three_mode:
        config.update({
            'online': True,
            'model': utils.load_model("REV_SpikingAE_opt.pth", num_inputs=800, num_outputs=800, num_hidden=384),
            'model_class': None, #utils.load_model_AE("NN_AE_opt_2.pth", num_inputs=800, num_outputs=800, num_hidden=384),
            'seed': 17,
            'scene_path': '/home/nino/PhD/Spiking/PotentialField_Sim/scenes/REV_potential_fields_sim_comparison_3.ttt',
            'save_path': 'simulation_data/REV_simulation_data_comparison_Literature',
            'pf_class': {
                'base': PF_REGISTRY[args.pf_base],
                'twin': PF_REGISTRY[args.pf_twin],
                'class': PF_REGISTRY[args.pf_class_mode],
            },
        })
    elif args.comparison:
        config['pf_class'] = {
            'base': PF_REGISTRY[args.pf_base],
            'twin': PF_REGISTRY[args.pf_twin],
        }
    else:
        config['pf_class'] = PF_REGISTRY[args.pf_base]
    return config


def save_robot_results(results, run_path, suffix=""):
    utils.plot_trajectory(results['pioneer_positions'], results['block_positions'],
                          results['goal_position'], results['preds'],
                          save=True, path=os.path.join(run_path, f"trajectory{suffix}.png"))
    utils.plot_thr(results['thresholds'], save=True,
                   path=os.path.join(run_path, f"THR{suffix}.png"))
    utils.plot_speed(results['speeds'], save=True,
                     path=os.path.join(run_path, f"speed{suffix}.png"))
    utils.plot_tot(results['pioneer_positions'], results['preds'], results['thresholds'],
                   results['speeds'], results['arrival_frame'],
                   save=True, path=os.path.join(run_path, f"summary{suffix}.png"))
    np.savez_compressed(os.path.join(run_path, f"simulation_data_dict{suffix}.npz"), **results)


def zip_frames(run_path, run, suffix=""):
    frames_path = os.path.join(run_path, f"frames_run{run}{suffix}")
    if os.path.exists(frames_path):
        shutil.make_archive(frames_path, 'zip', frames_path)
        shutil.rmtree(frames_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", type=int, required=True,
                        help="Number of simulation runs")
    parser.add_argument("--test", action="store_true",
                        help="Testing with online loop")
    parser.add_argument("--comparison", action="store_true",
                        help="Test comparison between online and offline")
    parser.add_argument("--three_mode", action="store_true",
                        help="Test comparison between online, offline and classical AE")
    parser.add_argument("--pf_base", choices=list(PF_REGISTRY), default='PotentialField',
                        help="PF class of the base robot")
    parser.add_argument("--pf_twin", choices=list(PF_REGISTRY), default='PotentialField',
                        help="PF class of the twin robot (comparison/three_mode)")
    parser.add_argument("--pf_class_mode", choices=list(PF_REGISTRY), default='PotentialField',
                        help="PF class of the class robot (three_mode)")
    parser.add_argument("--class_pf_only", action="store_true",
                        help="three_mode: the class robot uses only the selected PF class, without AE inference (like twin)")
    parser.add_argument("--moving_obstacles", action="store_true",
                        help="Slightly move the obstacles at every simulation step")
    parser.add_argument("--obstacle_turn", type=float, default=0.15,
                        help="Max random turn [rad] applied per step to each moving obstacle's preferred direction")
    parser.add_argument("--obstacle_every", type=int, default=10,
                        help="Move each obstacle one step every N frames (1 = move every frame)")
    parser.add_argument("--obstacle_goal_margin", type=float, default=2.5,
                        help="Minimum distance an obstacle must keep from the goal; on violation it immediately changes direction")
    args = parser.parse_args()

    config = setup_config(args)
    np.random.seed(config['seed'])
    seeds = np.random.randint(1, 1000, size=args.num_run)
    seeds = [144, 391]
    os.makedirs(config['save_path'], exist_ok=True)
    print("Starting...")

    suffixes = ['', '_twin', '_class']
    for decay in [0.1]:
        for i, run in enumerate(tqdm(range(args.num_run), desc="Simulation Runs")):
            run_path = os.path.join(config['save_path'],
                                    f"Run_{run:04d}_Seed_{seeds[i]}_rate_{str(decay).replace('.', '_')}")
            os.makedirs(run_path, exist_ok=True)

            simulator = Simulator(
                num_run=run,
                scene_path=config['scene_path'],
                seed=seeds,
                num_blocks=20,
                min_distance=2.5,
                min_goal_distance=8.0,
                save_path=run_path,
                online=config['online'],
                model=config['model'],
                THR_base=config['THR_base'],
                comparison=args.comparison,
                three_mode=args.three_mode,
                model_class=config['model_class'],
                THR_base_class=config['THR_base_class'],
                decay_rate=decay,
                decay_factor=config['decay_factor'],
                pf_class=config['pf_class'],
                class_pf_only=args.class_pf_only,
                moving_obstacles=args.moving_obstacles,
                obstacle_turn=args.obstacle_turn,
                obstacle_every=args.obstacle_every,
                obstacle_goal_margin=args.obstacle_goal_margin,
            )

            results_list = simulator.run()
            if not isinstance(results_list, tuple):
                results_list = (results_list,)

            for suffix, results in zip(suffixes, results_list):
                save_robot_results(results, run_path, suffix)
                zip_frames(run_path, run, suffix)

            preds = results_list[0]['preds']
            anomalies = np.bincount(preds, minlength=2)[-1].item()
            normal = np.bincount(preds, minlength=2)[0].item()
            utils.send_telegram(f"Run {i} completata con {anomalies} anomalie "
                                f"su {normal + anomalies} totali, rate {decay}")


if __name__ == "__main__":
    main()