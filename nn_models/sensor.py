import torch
from nn_models.encoder import Encoder
from nn_models.quantizer import VectorQuantizerEMA
import torchvision.transforms as T
import numpy as np

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
        return quantized.reshape(1,-1)