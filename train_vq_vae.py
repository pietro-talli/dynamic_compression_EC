import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from models import Encoder, Decoder
from codebook import Codebook
from torch.utils.data import Dataset, DataLoader
from utilities_dataset import create_dataset, ToTensor, FramesDataset
import argparse

path = sys.argv[1]

if not os.path.exists(path):
    os.mkdir(path)
    os.mkdir(path+'/images')
    create_dataset(50000, path)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(path)

embedding_dim = 8 #size of the vectors to be quantized
num_embeddings = 256 #number of codewords in the dictionary

if torch.cuda.is_available():
    device = 'cuda'
else: 
    device = 'cpu'

batch_size = 128

# create the dataset
dataset = FramesDataset(path+'/description.csv', path+'/images')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)

# create the encoder and decoder network
encoder = Encoder(embedding_dim)
decoder = Decoder(embedding_dim)

# vq_layer
vq_layer = Codebook(num_embeddings, embedding_dim, codebook_usage_threshold=5)

encoder.to(device)
decoder.to(device)
vq_layer.to(device)

# define the optimizer 
params_to_optimize = [{'params': encoder.parameters()},
                      {'params': decoder.parameters()},
                      {'params': vq_layer.parameters()}
                      ]
lr = 3e-4
optimizer = torch.optim.Adam(params_to_optimize, lr = lr)

committment_cost = 0.25

# training loop
train_res_recon_error = []
train_res_perplexity = [] 

num_epochs = int(sys.argv[2])
for i in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        input_tensor = batch.to(device)

        encoded_input = encoder(input_tensor)
        cb_output = vq_layer(encoded_input)
        
        quantized_input = cb_output.quantized
        perplexity = cb_output.perplexity

        input_reconstructed = decoder(quantized_input)
        
        error = 0.5*(input_reconstructed - input_tensor)**2
        recon_loss = error.mean()
        vq_loss = F.mse_loss(encoded_input, quantized_input.detach())
        loss = recon_loss + committment_cost*vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_res_recon_error.append(loss.item())
        train_res_perplexity.append(perplexity.item())

    print('%d iterations' % (i+1))
    print('recon_error: %.5f' % np.mean(train_res_recon_error[-100:]))
    print('perplexity: %.5f' % np.mean(train_res_perplexity[-100:]))
    print()
    writer.add_scalar('Loss/train', np.mean(train_res_recon_error[-100:]), i)
    writer.add_scalar('Codebook/perplexity', np.mean(train_res_perplexity[-100:]), i)

    torch.save(encoder.state_dict(), path+'/encoder_train.pt')
    torch.save(vq_layer.state_dict(), path+'/vq_layer_train.pt')      
    torch.save(decoder.state_dict(), path+'/decoder_train.pt')

encoder.to('cpu')
vq_layer.to('cpu')
decoder.to('cpu')

torch.save(encoder.state_dict(), path+'/encoder.pt')
torch.save(vq_layer.state_dict(), path+'/vq_layer.pt')      
torch.save(decoder.state_dict(), path+'/decoder.pt')