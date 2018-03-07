# easy_to_start-NCE-RNN-for-Pytorch
A basic framework of the Noise Contrastive Estimation (NCE) on RNN model. Can be run directly on PC and MAC


This RNN model is implemented with the Noise Contrastive Estimation (NCE) to substitute the Softmax. The basic idea follows the Ruder’s blog and Dyer (2014): sample noises from a prior model and train the RNN to distinguish the target from the noise.

Basic parameters:
nhid, embed = 250 and tied
nlayer = 1
Learning rate = 8
epochs = 2

Also, I use adaptive learning rate, which would decrease every 100 batches. Since the program will be early stopped, so adjusting learning rate per some batches will be useful than per epoch. In this model, the learning decreases 40% per epoch, and x0.92 per 100 batches.

For the NCE implementation, I add two new hyperparamters:  1. noise_ratio and 2. Z_offset. 

The noise ratio is the number of noise samples from a prior distribution. This model uses 25, which is claimed to be enough for approximating Softmax (Ruder’s blog). The prior model here is the unigram model based on the frequency of the train data. 

The second hyperparameter Z is original the denominator of the softmax. Some papers just consider Z as one, but in this model it deviates from 1. Hence, the Z is needed to tune as a hyperparameter. Meanwhile, the model will update the Z every 100 batches. The updated Z is calculated from an actual Softmax.

Here the best initial Z in this model is found out to be 9.5.

Also, a SGD with momentum = 0.9 is used to descent the parameter.
