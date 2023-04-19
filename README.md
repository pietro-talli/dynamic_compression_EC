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

## Train the policy with fixed quantization

After the AE has been trained it is possible to train the policy of the receiver (using a fixed quantization strategy at the transmitter). Use `train_policy.py` to train the policy. To run the function requires that in the `models` folder there are:
* `encoder.pt`
* `quantizer_K.pt` (a quantizer with the desired number of codewords)

example of how to use the function `train_policy.py`:
```
python train_policy.py --num_codewords 64
```
This command will train a policy using the `quantizer_64.pt` as the quantizer at the sensor side. The policty will be saved in the models folder as `policy_64.pt`

## Train the regressor to obtain semantic performance

