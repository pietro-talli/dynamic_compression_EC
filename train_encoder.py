import sys
import os 
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from nn_models.encoder import Encoder
from nn_models.quantizer import VectorQuantizerEMA
from nn_models.decoder import Decoder
from utilities_dataset import create_dataset, ToTensor, FramesDataset
from torch.utils.data import DataLoader
import numpy as np

num_training_updates = 50
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 8
num_embeddings = 64

#The input shape of the encoder is BCHW in this case (B,2,160,360)
#The output shape is (B,embedding_dim,2,4)
encoder = Encoder(2, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)

#Receives in input the encoded frames with shape (B,embedding_dim,2,4)
quantizer = VectorQuantizerEMA(num_embeddings,embedding_dim)

#The input shape is (B,embedding_dim,2,4)
decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

#Create a dataset
# Collect a dataset of tuples (o_t, o_{t+1})
collect_dataset = False
if collect_dataset:
    num_samples = 50000 #number of tuples to collect
    create_dataset(num_samples)

# Create a dataset
dataset = FramesDataset('dataset/description.csv', 'dataset/images', ToTensor())
dataloader = DataLoader(dataset, batch_size=128,
                        shuffle=True, num_workers=6)

writer = SummaryWriter()

# Create a dataset
dataset = FramesDataset('dataset/description.csv', 'dataset/images', ToTensor())
dataloader = DataLoader(dataset, batch_size=128,
                        shuffle=True, num_workers=6)

writer = SummaryWriter()

#Parameters of the VQ-VAE

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using ' + device)

param_to_optimize = [
                {'params': encoder.parameters()},
                {'params': quantizer.parameters()},
                {'params': decoder.parameters()}]
learning_rate = 1e-3
optimizer = torch.optim.Adam(param_to_optimize, lr=learning_rate, amsgrad=False)
batch_size = 128
_, h,w = dataset[0]['curr'].shape

encoder.train()
quantizer.train()
decoder.train()

train_res_recon_error = []
train_res_perplexity = []

reset = True
time_step = 0 


for i in range(num_training_updates):
    for i_batch, sample_batched in enumerate(dataloader):
        
        input_tensor = torch.cat((sample_batched['curr'], sample_batched['next']), dim = 1)
        input_tensor = input_tensor.to(device)
        data = 1-input_tensor
        optimizer.zero_grad()

        z_e = encoder(data)
        vq_loss, quantized, perplexity, _ = quantizer(z_e, reset)
        data_recon = decoder(quantized) 
        recon_error = F.mse_loss(data_recon, data)
        loss = recon_error + vq_loss
        loss.backward()
        optimizer.step()
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())
        reset = False
        
        if (i_batch+1) % 50 == 0:
            print('%d iterations' % (i+1))
            print('recon_error: %.5f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.5f' % np.mean(train_res_perplexity[-100:]))
            print()
            reset = True #Reset the unused codewords every 50 iteroations
            
            writer.add_scalar('Loss/train', np.mean(train_res_recon_error[-100:]), time_step)
            writer.add_scalar('Perplexity/batch', np.mean(train_res_perplexity[-100:]), time_step)
            time_step += 1
            torch.save(encoder.state_dict(), 'models/encoder_training.pt')
            torch.save(quantizer.state_dict(), 'models/quantizer_'+str(num_embeddings)+'_training.pt')
            torch.save(decoder.state_dict(), 'models/decoder_training.pt')

torch.save(encoder.state_dict(), 'models/encoder.pt')
torch.save(quantizer.state_dict(), 'models/quantizer_'+str(num_embeddings)+'.pt')
torch.save(decoder.state_dict(), 'models/decoder.pt')