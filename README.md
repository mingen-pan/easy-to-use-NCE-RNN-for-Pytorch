# easy_to_start-NCE-RNN-for-Pytorch
A basic framework of the Noise Contrastive Estimation (NCE) on RNN model. Can be run directly on PC and MAC



This RNN model is implemented with the Noise Contrastive Estimation (NCE) to substitute the Softmax. The basic idea follows the Ruder’s blog and Dyer (2014): sample noises from a prior model and train the RNN to distinguish the target from the noise.

This is a beginner-friendly model. The NCE loss-function module is located in the nce.py, which can be directly implemented into your own network. However, I include no GPU boosting code here, so users may need to boost them by themselves.


Initialize the criterion module,

`train_criterion = NCELoss(Q, noise_ratio, Z_offset)`


Run the RNN model and get the output:

`output, hidden = model(data, hidden)`

Then, convert the output into 2D tensor and run the NCE loss

`loss = train_criterion(output.view(-1, ntokens), targets)`

Finally, back-propopagate through the Network:

`loss.backward()`

The NCEloss in the nce.py would receive a output tensor from the network and the corresponding target. The output tensor should be 2D tensor (B' x N) and the target tensor should have the size of B' in 1D or B' x 1 in 2D. Generally the RNN would output a 3D tensor like (B x L x N), please convert into (BL x N). Similarly convert the target (B x L) into (BL in 1D). Luckily, in my code, even you don't do the conversion, the code still works.

The B, L, N are the batch size, sequence length, and total vocabulary size.

For the NCE implementation, I use two new hyperparamters:  1. noise_ratio and 2. Z_offset. 

The noise ratio is the number of noise samples from a prior distribution. This model uses 25, which is claimed to be enough for approximating Softmax (Ruder’s blog). The prior model here is the unigram model based on the frequency of the train data. 

The second hyperparameter Z is original the denominator of the softmax. Some papers just consider Z as one, but in this model it deviates from 1. Hence, the Z is needed to tune as a hyperparameter. Meanwhile, the model will update the Z every 100 batches. The updated Z is calculated from an actual Softmax.

Here the best initial Z in this model is found out to be 9.5.

------------------------------------------------------------------------------------------------------------------------------
The following context is about the RNN model here rather than the NCE loss.

Also, I use adaptive learning rate, which would decrease every 100 batches. Since the program will be early stopped, so adjusting learning rate per some batches will be useful than per epoch. In this model, the learning decreases 40% per epoch, and x0.92 per 100 batches.

In addition, a SGD with momentum = 0.9 is used to descent the parameter.

------------------------------------------------------------------------------------------------------------------------------
The RNN model is from: https://github.com/pytorch/examples/tree/master/word_language_model

I have done some modification on it: use SCD with momentum optimizer, add the learning rate decay through batches and epoches, and the automatically update Z_offset from Softmax every 100 batches.
