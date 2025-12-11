import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from snnae import SAE

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

    @staticmethod
    def acquire_image(path):
        return cv2.imread(path,0)
    
    @staticmethod
    def preprocess_2828(image):
        return cv2.resize(image, (28, 28))
    
    @staticmethod
    def to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def plot_tot(pioneer_positions, block_positions, goal_position, preds):
        plt.figure(figsize=(10, 10))
        plt.title("Traiettorie Pioneer con goal e blocchi")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.grid(True)
        plt.axis('equal')
        block_size = 1.0
        plt.xticks(np.arange(8.5, -8.5, -1))
        plt.yticks(np.arange(8.5, -8.5, -1))
        plt.plot(pioneer_positions[0][0], pioneer_positions[0][1] , 'ro', markersize=12, label='Start')

        x_path = [pos[0] for pos in pioneer_positions[1:]]
        y_path = [pos[1] for pos in pioneer_positions[1:]]
        plt.plot(x_path, y_path, '-', label='Path')
        anom_idx = np.where(preds[:] == 1)[0]
        if len(anom_idx) > 0:
                plt.plot(np.array(x_path)[anom_idx], np.array(y_path)[anom_idx], 'ro', label='Anomalie')

        # Goal
        plt.plot(goal_position[0], goal_position[1] , 'go', markersize=12, label='Goal')

        for i in range(len(block_positions)):
                x, y = block_positions[i]
                rect = plt.Rectangle((x - block_size/2, y - block_size/2), block_size, block_size,
                                linewidth=1, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                plt.plot(x, y, 'r.', markersize=5)
        plt.grid(False)
        plt.legend()
        plt.show()