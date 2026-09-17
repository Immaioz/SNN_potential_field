import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import requests
from snnae import SAE, AE
from config import BOT_TOKEN, CHAT_ID

class utils:
    # @staticmethod
    # def preprocess(image):
    #     return cv2.resize(image, (512, 512))

    # @staticmethod
    # def low_res(image):
    #     return utils.preprocess(image)[0::12, 0::12]

    # @staticmethod
    # def preprocess_predict(image):
    #     return np.expand_dims(utils.preprocess(tf.keras.applications.vgg19.preprocess_input(image)), axis=0)



    def load_model(path, num_inputs=800, num_hidden=500, num_outputs=800, num_steps=25, beta=0.95):
        model = SAE(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs, num_steps=num_steps, beta=beta)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # self.load_state_dict(torch.load(path))
        return model
    

    def load_model_AE(path, num_inputs=800, num_hidden=500, num_outputs=800):
        model = AE(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # self.load_state_dict(torch.load(path))
        return model

    @staticmethod
    def acquire_image(path):
        return cv2.imread(path,0)
    
    @staticmethod
    def preprocess_2828(image):
        return cv2.resize(image, (28, 28))
    
    @staticmethod
    def to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def plot_trajectory(pioneer_positions, block_positions, goal_position, preds, save=False, path=None, robot_size=(0.5, 0.4), robot_every=30, sensor_values=None, obstacles_hit=None):
        plt.figure(figsize=(10, 10))
        plt.title("Traiettorie Pioneer con goal e blocchi")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.grid(True)
        plt.axis('equal')
        block_size = 0.5
        plt.xticks(np.arange(8.5, -8.5, -1))
        plt.yticks(np.arange(8.5, -8.5, -1))

        # Start
        plt.plot(pioneer_positions[0][0], pioneer_positions[0][1], 'bo', markersize=12, label='Start')

        # Path 
        x_path = np.array([pos[0] for pos in pioneer_positions[1:]])
        y_path = np.array([pos[1] for pos in pioneer_positions[1:]])
        if sensor_values is not None:
            sv = np.asarray(sensor_values)
            if sv.ndim == 1 and len(sv) == len(pioneer_positions):
                colors = sv[1:]
            else:
                colors = sv
            sc = plt.scatter(x_path, y_path, c=colors, cmap='viridis', s=25,
                             edgecolors='none', label='Distanza sensore min')
            plt.colorbar(sc, ax=plt.gca(), label='Min distanza sensore [m]')
        else:
            plt.plot(x_path, y_path, '-', label='Path')

        # Robot footprint (lateral sides only)
        W, H = robot_size
        positions = np.asarray(pioneer_positions)
        n = len(positions)
        step = max(1, n // robot_every) if n > 1 else 1
        for i in range(0, n, step):
            if i == 0:
                a, b = 0, min(1, n - 1)
            elif i == n - 1:
                a, b = i - 1, i
            else:
                a, b = i - 1, i + 1
            angle = np.arctan2(positions[b][1] - positions[a][1],
                               positions[b][0] - positions[a][0])
            fw = np.array([np.cos(angle), np.sin(angle)])
            lt = np.array([-np.sin(angle), np.cos(angle)])
            p = positions[i]
            for side in (-1, 1):
                s0 = p + (H / 2) * fw + side * (W / 2) * lt
                s1 = p - (H / 2) * fw + side * (W / 2) * lt
                plt.plot([s0[0], s1[0]], [s0[1], s1[1]], 'b-', linewidth=1)

        # Anomalie
        anom_idx = np.where(preds[:] == 1)[0] - 1  
        anom_idx = anom_idx[(anom_idx >= 0) & (anom_idx < len(x_path))] 
        if len(anom_idx) > 0:
            plt.plot(x_path[anom_idx], y_path[anom_idx], 'ro', label='Anomalie')

        # Obstacles hit
        if obstacles_hit is not None:
            obst_idx = np.where(np.asarray(obstacles_hit) == 1)[0] - 1
            obst_idx = obst_idx[(obst_idx >= 0) & (obst_idx < len(x_path))]
            if len(obst_idx) > 0:
                plt.plot(x_path[obst_idx], y_path[obst_idx], 'ms', markersize=6,
                         label='Obstacle hit')

        # Goal
        plt.plot(goal_position[0], goal_position[1], 'go', markersize=12, label='Goal')

        # Blocchi
        for x, y in block_positions:
            rect = plt.Rectangle((x - block_size / 2, y - block_size / 2),
                                block_size, block_size,
                                linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            plt.plot(x, y, 'r.', markersize=5)

        plt.grid(False)
        plt.legend()
        plt.tight_layout()

        if save:
            plt.savefig(path)
        else:
            plt.show()

        plt.close()


    def plot_thr(thr, save=False, path=None):
        plt.figure(figsize=(10,10))
        plt.plot(thr)
        plt.title("Threshold decay")
        plt.xlabel("Steps")
        plt.ylabel("Threshdold")
        plt.tight_layout()
        if save:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()


    def plot_speed(speed, save=False, path=None):
        plt.figure(figsize=(10,10))
        plt.plot(speed)
        plt.title("Speed evolution")
        plt.xlabel("Steps")
        plt.ylabel("Speed")
        plt.tight_layout()
        if save:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    def plot_tot(pioneer_positions, preds, thr, speed, arrival, save=False, path=None):
        fig, axes = plt.subplots(
            3, 1,
            figsize=(10, 10),
            gridspec_kw={'height_ratios': [0.3, 0.3, 0.3]}
        )

        
        frames = np.arange(len(pioneer_positions)-1)
        y_line = np.zeros_like(frames)
        x_ticks = np.arange(0, len(pioneer_positions), step=100)
        
        ax = axes[0]
        ax.plot(frames, y_line, '-', color='black', alpha=0.5)
        if arrival is not None:
            ax.axvline(arrival+1, color='r', linestyle='--', label='Arrival Frame')
        anom_idx = np.where(preds == 1)[0]
        ax.plot(anom_idx, np.zeros_like(anom_idx), 'ro')
        ax.set_yticks([])
        ax.set_xticks(x_ticks)
        ax.set_xlabel("Steps")
        ax.set_title("Anomaly timeline")
        ax.set_title("Anomaly events along trajectory")

        ax = axes[1]
        if isinstance(thr, float):
            thr = np.full(len(pioneer_positions), thr)
        ax.plot(thr)
        ax.set_title("Threshold decay")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Threshold")
        ax.set_xticks(x_ticks)
        if arrival is not None:
            ax.axvline(arrival+1, color='r', linestyle='--', label='Arrival Frame')

        ax = axes[2]
        ax.plot(speed)
        ax.set_title("Speed evolution")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Speed")
        ax.set_xticks(x_ticks)
        if arrival is not None:
            ax.axvline(arrival+1, color='r', linestyle='--', label='Arrival Frame')

        plt.tight_layout()
        if save:
            plt.savefig(path)
        else:
            plt.show()

        plt.close()

    def send_telegram(msg):
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg}
        requests.post(url, data=payload)