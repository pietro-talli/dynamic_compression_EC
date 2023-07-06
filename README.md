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

To train a regressor at the receiver to recontruct the physical state of the system, the script `train_regerssor.py` can be used as follows

```
python train_regeressor.py --num_codewords 64
```
This will train and save an object of type `PhysicalValueRegressor` which implements Pytorch RNN. 

## Obtaining 3 Levels of communication

At this point, we can learn a policy at the Sensor side in order to select the most suitable quantizer to ecode the current observation. Note that the transmitter policy can be trained in order to maximize three different performance metrics at the receiver which correspond respectively to:
* Technical problem (Level A)
* Semantic problem (Level B)
* Effectiveness problem (Level C)

use the `train_sensor.py` script to train a policy at the sensor side. Here is an example of how to use it:
```
python train_sensor.py \
--num_episodes 100000 \
--level C \
--beta 0.1
```
In this case, the sensor policy will be trained (for 100000 episodes) to maximize the performance at Level C using a trade-off parameter $\beta = 0.1$.

The models will be saved in the `models` folder as `sensor_policy_levelC_beta0.1.pt`. 

## Test the communication system

The performance plots can be obtained by testing each level with each $\beta$ value. It is suggested to test the system averaging at least over 500 episodes to reduce variance due to random initializations.

To obtain the colormaps which describe choices of the transmitter policy with respect to the control actions at the reciever, use the script `plot_gradients.py` as follows:
```
python plot_gradient.py \
--retrain True
```
changing the name of the model that is loaded in the script with the one to be used for testing.

To obtain the plot of the sensor action distribution with respect to the entropy of the poloicy at the receiver use the script `plot_entropy.py` as follows:
```
python plot_entropy.py --t_lag 3
```
this command will produce the colormap corresponding to the `t_lag` time specified (in this case 3). 

## Further details
All the scripts are based on the python `gym` library and the `pytorch` library for the neural networks. In principle it is possible to use any environment that is compatible with the `gym` library. All the training scripts should work straight away. However, the plots scripts might require some changes in order to work with different environments. 

All the models are defined in the `nn_models` folder and can be easily modified. 

In `utils/rl_utils.py` it is possible to implement other RL algorithms (e.g. PPO, DDPG, etc.) to train the policies. Two algorithms are already implemented:
* Deep Q-Learning
* Advantage Actor Critic (A2C)
