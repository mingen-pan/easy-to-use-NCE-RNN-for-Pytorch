import torch

def Q_from_tokens(token_list, dictionary):
    freq = [0]*len(dictionary)
    for idx in token_list:
        freq[idx] +=1
    Q = torch.FloatTensor(freq)
    return Q/Q.sum()

def adjust_learning_rate(optimizer, decay):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decay)
        lr_updated = param_group['lr']
    return lr_updated