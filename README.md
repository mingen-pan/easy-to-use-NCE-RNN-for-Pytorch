# easy-to-use-NCE-RNN-for-Pytorch
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

<br/>
<br/>

Time Saving:

Given a LSTM RNN model with with one layer of 200 hidden units and the batch size of 35*32, using the Penn Tree Bank database(10000 words), the average time of the forward propagation from the Softmax and the back propagation from Softmax to each parameters is followed:

```
t_s = time.time()
loss = train_criterion(output.view(-1, ntokens), targets)
loss.backward()  
softmax_time += time.time() - t_s
 ```
  
The result is:
 
```
0.1559 seconds for NCE method
0.3797 seconds for ordinary Softmax method
```

Therefore, each NCE can save 0.22 seconds each time, more than 50% of the total back propagation time.

<br/>
<br/>

Performance:

NCE:

`| end of epoch   1 | time: 220.15s | valid loss  5.24 | valid perplexity   188.26`

Ordinary Softmax:

`| end of epoch   1 | time: 406.43s | valid loss  5.15 | valid perplexity   172.76`

They are more or less similar. Of course, the ordinary softmax should behave a bit better.

------------------------------------------------------------------------------------------------------------------------------
The following context is about the RNN model here rather than the NCE loss.

Also, I use adaptive learning rate, which would decrease every 100 batches. Since the program will be early stopped, so adjusting learning rate per some batches will be useful than per epoch. In this model, the learning decreases 40% per epoch, and x0.92 per 100 batches.

In addition, a SGD with momentum = 0.9 is used to descent the parameter.

------------------------------------------------------------------------------------------------------------------------------
The RNN model is from: https://github.com/pytorch/examples/tree/master/word_language_model

I have done some modification on it: use SCD with momentum optimizer, add the learning rate decay through batches and epoches, and the automatically update Z_offset from Softmax every 100 batches.
