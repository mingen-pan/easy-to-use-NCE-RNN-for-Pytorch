import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class NCELoss(nn.Module):
    def __init__(self, Q, noise_ratio, Z_offset = 9.5):
        super(NCELoss, self).__init__()
        assert torch.is_tensor(Q), "the Q should be a tensor"
        assert type(noise_ratio) is int
        # the Q is prior model
        # the N is the number of vocabulary
        # the K is the noise sampling times
        self.Q = Variable(Q)
        self.N = Q.size(0)
        self.K = noise_ratio
        # exp(X)/Z = exp(X-Z_offset) 
        self.Z_offset = Z_offset

    def update_Z_offset(self, new_Z):
        self.Z_offset = math.log(new_Z)

    def forward(self, output, target):
        #output is the RNN output, which is the input of loss function
        output = output.view(-1, self.N)
        B = output.size(0)
        noise_idx = self.get_noise(B)
        idx = self.get_combined_idx(target, noise_idx)
        P_target, P_noise = self.get_prob(idx, output, sep_target = True)
        Q_target, Q_noise = self.get_Q(idx)
        loss = self.nce_loss(P_target, P_noise, Q_noise, Q_target)
        return loss.mean()
        
        
    def get_Q(self, idx, sep_target = True):
        idx_size = idx.size()
        prob_model = self.Q[idx.view(-1)].view(idx_size)
        if sep_target:
            #target, noise
            return prob_model[:,0], prob_model[:,1:]
        else:
            return prob_model
        
    def get_prob(self, idx, scores, sep_target = True):
        scores = self.get_scores(idx, scores)
        prob = scores.sub(self.Z_offset).exp()
        if sep_target:
            #target, noise
            return prob[:,0], prob[:,1:]
        else:
            return prob
    def get_scores(self, idx, scores):
        B, N = scores.size()
        #the K = self.K + 1
        K = idx.size(1)
        idx_increment = N * torch.arange(B).view(B,1) *torch.ones(K)
        idx_increment = Variable(idx_increment.long(), requires_grad = False)
        new_idx = idx_increment + idx
        new_scores = scores.view(-1).index_select(0, new_idx.view(-1))
        return new_scores.view(B, K)
    
    def get_noise(self, batch_size, uniform = False):
        # this function would also convert the target into (-1, N)
        if uniform:
            noise = np.random.randint(self.N, size = self.K * batch_size)
        else:
            noise = np.random.choice(self.N, self.K * batch_size, replace= True, p = self.Q.data )
        noise_idx = Variable(torch.LongTensor(noise).view(batch_size, self.K))
        return noise_idx

    def get_combined_idx(self, target_idx, noise_idx):
        return torch.cat((target_idx.view(-1,1), noise_idx), 1)
    
    def nce_loss(self, prob_model, prob_noise_in_model, prob_noise, prob_target_in_noise):
        # prob_model: P_target
        # prob_target_in_noise: Q_target
        # prob_noise_in_model: P_noise
        # prob_noise: Q_noise
        def safe_log(tensor):
            EPSILON = 1e-10
            return torch.log(EPSILON + tensor)

        model_loss = safe_log(prob_model / (
            prob_model + self.K * prob_target_in_noise
        )).view(-1)

        noise_loss = torch.sum(
            safe_log((self.K * prob_noise) / (prob_noise_in_model + self.K * prob_noise)), -1
        ).view(-1)

        loss = - (model_loss + noise_loss)

        return loss