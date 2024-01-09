from typing import Any
import torch
from nn_models.encoder import Encoder
from nn_models.quantizer import VectorQuantizerEMA
import torchvision.transforms as T
import numpy as np
import torch.nn as nn
import cv2
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render().transpose((2, 0, 1))
    
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8), int(screen_width*0.2):int(screen_width*0.8)]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    
    screen = T.ToPILImage()(screen)
    screen = screen.convert('L')
    #screen = T.Resize(80, interpolation=T.InterpolationMode.BICUBIC)(screen)
    screen = T.ToTensor()(screen)
    return screen

class Sensor(nn.Module):
    def __init__(self, encoder: Encoder, quantizer: VectorQuantizerEMA):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer

    def forward(self, env, prev_screen=None):
        if prev_screen is None:
            prev_screen = get_screen(env)
        curr_screen = get_screen(env)
        h,w = curr_screen.squeeze().numpy().shape
        input_tensor = torch.reshape(torch.cat((prev_screen, curr_screen)), (1,2,h,w))
        encoded = self.encoder(1-input_tensor)
        _,quantized,_,_ = self.quantizer(encoded)
        return quantized.reshape(1,-1), curr_screen
    
class Sensor_not_quantized(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, env, prev_screen=None):
        if prev_screen is None:
            prev_screen = get_screen(env)
        curr_screen = get_screen(env)
        h,w = curr_screen.squeeze().numpy().shape
        input_tensor = torch.reshape(torch.cat((prev_screen, curr_screen)), (1,2,h,w))
        encoded = self.encoder(1-input_tensor.to(device))
        return encoded, curr_screen
    
class Sensor_not_quantized_level_A(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, env, prev_screen=None):
        if prev_screen is None:
            prev_screen = get_screen(env)
        curr_screen = get_screen(env)
        h,w = curr_screen.squeeze().numpy().shape
        input_tensor = torch.reshape(torch.cat((prev_screen, curr_screen)), (1,2,h,w))
        encoded = self.encoder(1-input_tensor.to(device))
        return encoded, curr_screen, 1-input_tensor

class SensorDigital():
    def __init__(self, h,w):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h = h
        self.w = w

    def to(smt):
        pass

    def __call__(self, env, prev_screen = None) -> Any:
        screen = get_screen(env)
        screen = screen.numpy()
        screen = screen.transpose(1,2,0)
        screen = screen[:,:,0]

        # cast to uint8
        gray = screen.astype('uint8')

        # threshold the grayscale image
        thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # resize the image
        thresh = cv2.resize(thresh, (self.h,self.w))

        cv2.imwrite('temp.png', thresh, [cv2.IMWRITE_PNG_COMPRESSION, 100])
        compressed = cv2.imread('temp.png', 0)

        compressed = compressed/255.
        compressed = compressed.astype('float32')
        compressed = torch.from_numpy(compressed)
        compressed = compressed.to(self.device)
        # resize the image
        thresh = cv2.resize(thresh, (self.h,self.w))

        return compressed.reshape(1,-1), screen
        