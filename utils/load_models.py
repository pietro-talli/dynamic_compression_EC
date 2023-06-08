import torch
import torch.nn as nn

from nn_models.encoder import Encoder
from nn_models.decoder import Decoder
from nn_models.quantizer import VectorQuantizerEMA
from nn_models.policy import RecA2C
from nn_models.regressor import PhysicalValueRegressor

def load_models(name_agent_policy, name_sensor_policy):

    embedding_dim = 8
    num_feat = 8

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2

    encoder = Encoder(2, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)
    encoder.load_state_dict(torch.load('../models/encoder.pt', map_location=torch.device('cpu')))

    decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)
    decoder.load_state_dict(torch.load('../models/decoder.pt', map_location=torch.device('cpu')))

    quantizers = []
    num_quantization_levels = [0,1,2,3,4,5,6]
    num_codewords_s = [2,4,8,16,32,64]

    for i in num_quantization_levels[1:]:
        quantizer = VectorQuantizerEMA(num_codewords_s[i-1], embedding_dim)
        quantizer.load_state_dict(torch.load('../models/quantizer_'+str(num_codewords_s[i-1])+'.pt', map_location=torch.device('cpu')))
        quantizers.append(quantizer)
        
    list_of_quantizers = nn.ModuleList(quantizers)

    

    return encoder, decoder, list_of_quantizers, agent_policy, sensor_policy
