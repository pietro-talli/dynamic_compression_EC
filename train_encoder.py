import sys
import os 
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from nn_models.encoder import Encoder
from nn_models.quantizer import VectorQuantizerEMA
from nn_models.decoder import Decoder
from utils.utilities_dataset import create_dataset, ToTensor, FramesDataset
from torch.utils.data import DataLoader
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--num_samples', type=int, help='number of samples in the dataset', required=False)
parser.add_argument('--epochs', type=int, help='The number of epochs to train the AE', required=False)
parser.add_argument('--embedding_dim', type=int, help='The embedding dimension of the latent features', required=False)
parser.add_argument('--num_codewords', type=int, help='The number of codewords to use i the quantizer', required=False)
parser.add_argument('--retrain', type=bool, help='Set to False if the encoder-decoder already exist', required=False)

args = parser.parse_args()

num_samples = 50000 #number of tuples to collect
collect_dataset = False
if args.num_samples:
    num_samples = args.num_samples
    collect_dataset = True

num_training_updates = 100
embedding_dim = 64
num_embeddings = 64
retrain = True

if args.epochs:
    num_training_updates = args.epochs
if args.embedding_dim:
    embedding_dim = args.embedding_dim
if args.num_codewords:
    num_embeddings = args.num_codewords
if args.retrain:
    retrain = args.retrain

if num_embeddings != 64:
    retrain = False

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

#The input shape of the encoder is BCHW in this case (B,2,160,360)
#The output shape is (B,embedding_dim,2,4)
encoder = Encoder(2, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)

#Receives in input the encoded frames with shape (B,embedding_dim,2,4)
quantizer = VectorQuantizerEMA(num_embeddings,embedding_dim)

#The input shape is (B,embedding_dim,2,4)
decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

#Create a dataset
# Collect a dataset of tuples (o_t, o_{t+1})

if collect_dataset:
    if not os.path.exists('../dataset'):
        os.mkdir('../dataset')
        os.mkdir('../dataset/images')
    print('Collecting the dataset')
    create_dataset(num_samples, '../dataset')
    print('Dataset collected')

writer = SummaryWriter('../runs')


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Parameters of the VQ-VAE
param_to_optimize = [
                {'params': encoder.parameters()},
                {'params': quantizer.parameters()},
                {'params': decoder.parameters()}]

encoder.train()
quantizer.train()
decoder.train()

if not retrain:
    if os.path.exists('../models'):
        param_to_optimize = [{'params': quantizer.parameters()}]
        encoder.load_state_dict(torch.load('../models/encoder.pt'))
        decoder.load_state_dict(torch.load('../models/decoder.pt'))
        encoder.eval()
        decoder.eval()
#Create the models folder
if not os.path.exists('../models'):
    os.mkdir('../models')

#define the optimizer, learning rate and batch size
learning_rate = 1e-3
optimizer = torch.optim.Adam(param_to_optimize, lr=learning_rate, amsgrad=False)
batch_size = 128

# Create a dataset
dataset = FramesDataset('../dataset/description.csv', '../dataset/images', ToTensor())
dataloader = DataLoader(dataset, batch_size=128,
                        shuffle=True, num_workers=6)

# shape of the observation 
_, h,w = dataset[0]['curr'].shape

train_res_recon_error = []
train_res_perplexity = []

reset = True
time_step = 0 

print('using ' + device.type)
encoder.to(device)
quantizer.to(device)
decoder.to(device)

# Train the model

for i in range(num_training_updates):
    for i_batch, sample_batched in enumerate(dataloader):
        
        #The input tensor is of shape (B,2,160,360)
        input_tensor = torch.cat((sample_batched['curr'], sample_batched['next']), dim = 1)
        input_tensor = input_tensor.to(device)
        data = 1-input_tensor
        optimizer.zero_grad()

        #The output of the encoder is of shape (B,embedding_dim,2,4)
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
        
        #Print the loss every 100 iterations
        if (i_batch+1) % 100 == 0:
            print('%d iterations' % (i+1))
            print('recon_error: %.5f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.5f' % np.mean(train_res_perplexity[-100:]))
            print()
            reset = True #Reset the unused codewords every 100 iteroations

            writer.add_scalar('Loss/train', np.mean(train_res_recon_error[-100:]), time_step)
            writer.add_scalar('Perplexity/batch', np.mean(train_res_perplexity[-100:]), time_step)
            time_step += 1
            
            torch.save(quantizer.state_dict(), '../models/quantizer_'+str(num_embeddings)+'_training.pt')
            if retrain:
                torch.save(encoder.state_dict(), '../models/encoder_training.pt')
                torch.save(decoder.state_dict(), '../models/decoder_training.pt')


torch.save(quantizer.state_dict(), '../models/quantizer_'+str(num_embeddings)+'.pt')
if retrain:
    torch.save(encoder.state_dict(), '../models/encoder.pt')
    torch.save(decoder.state_dict(), '../models/decoder.pt')