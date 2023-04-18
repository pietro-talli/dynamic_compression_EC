# Code for Effective and Semantic communication

## Train the AutoEncoder

Use `train_encoder.py` to train an autoencdoer where the latent space is quantized using vector quantization 
```
python train_encoder.py \
--num_samples 50000 \
--epochs 100 \
--embedding_dim 64 \
--num_codewords 64 \
--retrain True 
```
* if argument `--num_samples` is given the script will collect a new dataset containing 2 $ \times$ `num_samples` images. 

* `--epochs` can be used to indicate the number of epochs to train the AE (default is 100)
* `--embedding_dim` can be used to choose the size of the latent features (default is 64)
* `--num_codewords` can be used to decide how many codewords there are in the codebook (default is 64)
* `--rertrain` can be set to `False` to avoid retraining (default is `True`) the `encoder` and `decoder` and just obtain a new quantizer 

Usage: 

use
```
python train_encoder.py \
--num_samples 50000 \
--epochs 100 \
--embedding_dim 64 \
--num_codewords 64 \
--retrain True 
```
the first time to create the dataset, train the `encoder`, the `quantizer` and the `decoder`. 

use 
```
python train_encoder.py \
--epochs 100 \
--embedding_dim 64 \
--num_codewords 32 \
--retrain False
```
to obtain a new quantizer with 32 codewords without retraining the `encoder` and the `decoder`.

Each run  `encoder`, `quantizer` and `decoder` will be saved in the folder `models` respectively as:
* `encoder.pt`
* `quantizer_K.pt`
* `decoder.pt`

where K = `num_codewords` for the quantizer selceted. 