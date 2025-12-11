import cv2
import numpy as np

class utils:
    @staticmethod
    def preprocess(image):
        return cv2.resize(image, (512, 512))

    @staticmethod
    def low_res(image):
        return utils.preprocess(image)[0::12, 0::12]

    # @staticmethod
    # def preprocess_predict(image):
    #     return np.expand_dims(utils.preprocess(tf.keras.applications.vgg19.preprocess_input(image)), axis=0)
    
    @staticmethod
    def acquire_image(path):
        return cv2.imread(path,0)
    
    @staticmethod
    def preprocess_2828(image):
        return cv2.resize(image, (28, 28))
    
    @staticmethod
    def to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)