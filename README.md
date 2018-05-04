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
softmax_time = 0
count = 0
for i in range(N):
    ### some training
    t_s = time.time()
    loss = train_criterion(output.view(-1, ntokens), targets)
    loss.backward()  
    softmax_time += time.time() - t_s
    count += 1
print(softmax_time/count)

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

<br/>
<br/>

Here is the result of running for 2 epochs:

(parameters are the default values)

```
| epoch   1 |   200/  829 batches | lr 6.62 | ms/batch 310.14 | loss  4.01 | perplexity    55.41
| epoch   1 |   400/  829 batches | lr 5.49 | ms/batch 313.78 | loss  3.46 | perplexity    31.90
| epoch   1 |   600/  829 batches | lr 4.54 | ms/batch 310.64 | loss  3.28 | perplexity    26.46
| epoch   1 |   800/  829 batches | lr 3.76 | ms/batch 302.40 | loss  3.20 | perplexity    24.59
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 270.69s | valid loss  5.18 | valid perplexity   177.23
-----------------------------------------------------------------------------------------
| epoch   2 |   200/  829 batches | lr 2.65 | ms/batch 320.44 | loss  3.09 | perplexity    22.04
| epoch   2 |   400/  829 batches | lr 2.19 | ms/batch 312.87 | loss  3.00 | perplexity    20.15
| epoch   2 |   600/  829 batches | lr 1.82 | ms/batch 319.28 | loss  2.92 | perplexity    18.45
| epoch   2 |   800/  829 batches | lr 1.50 | ms/batch 314.54 | loss  2.92 | perplexity    18.48
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 276.48s | valid loss  4.96 | valid perplexity   142.40
-----------------------------------------------------------------------------------------
=========================================================================================
| End of training | test loss  4.92 | test perplexity   136.82
=========================================================================================
```
------------------------------------------------------------------------------------------------------------------------------
The following context is about the RNN model here rather than the NCE loss.

Also, I use adaptive learning rate, which would decrease every 100 batches. Since the program will be early stopped, so adjusting learning rate per some batches will be useful than per epoch. In this model, the learning decreases 40% per epoch, and x0.92 per 100 batches.

In addition, a SGD with momentum = 0.9 is used to descent the parameter.

------------------------------------------------------------------------------------------------------------------------------
The RNN model is from: https://github.com/pytorch/examples/tree/master/word_language_model

I have done some modification on it: use SCD with momentum optimizer, add the learning rate decay through batches and epoches, and the automatically update Z_offset from Softmax every 100 batches.
