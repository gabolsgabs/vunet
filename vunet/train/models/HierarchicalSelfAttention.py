# -*- coding: utf-8 -*-

"""
Implementation of the hierarchical attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
#from sru import SRU


# Attention layer
class ChordLevelAttention(nn.Module):
    # this follows the word-level attention from Yang et al. 2016
    # https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
    def __init__(self, n_hidden, batch_first=True):
        super(ChordLevelAttention, self).__init__()
        self.mlp = nn.Linear(n_hidden, n_hidden)
        self.u_w = nn.Parameter(torch.rand(n_hidden))
        self.batch_first = batch_first

    def forward(self, X):
        if not self.batch_first:
            # make the input (batch_size, timesteps, features)
            X = X.transpose(1, 0)
        # get the hidden representation of the sequence
        u_it = F.tanh(self.mlp(X))
        # get attention weights for each timestep
        alpha = F.softmax(torch.matmul(u_it, self.u_w), dim=1)
        # get the weighted sum of the sequence
        out = torch.sum(torch.matmul(alpha, X), dim=1)
        return out, alpha

class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)

class BottleLinear(Bottle, Linear):
    ''' Perform the reshape routine before and after a linear projection '''
    pass

class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    pass

class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

class BatchBottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0]*size[1]))
        return out.view(-1, size[0], size[1])

#%%

class BottleLayerNormalization(BatchBottle, LayerNormalization):
    ''' Perform the reshape routine before and after a layer normalization'''
    pass

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1, average=-1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax()
        self.average = average

    def forward(self, q, k, v, attn_mask=None):
        if (q.size(1) != v.size(1)):
            q = q.repeat(1, v.size(1) // q.size(1), 1)
            k = k.repeat(1, v.size(1) // k.size(1), 1)
        if (k.dim() > 2):
            attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        else:
            attn = torch.bmm(q.unsqueeze(2), k.unsqueeze(2).transpose(1, 2)) / self.temper
        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask, -float('inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        if (v.dim() > 2):
            output = torch.bmm(attn, v)
        else:
            output = torch.bmm(attn, v.unsqueeze(2)).squeeze(2)
        if (self.average > 0):
            output = torch.mean(output, self.average).unsqueeze(self.average)
        return output, attn

class KernelAttention(nn.Module):
    ''' Kernel-Based Attention module '''
    def __init__(self, n_kernel, d_model, d_trans, average=-1, dropout=0.1):
        super(KernelAttention, self).__init__()
        # Parameters
        self.n_kernel = n_kernel
        self.average = average
        self.d_trans = d_trans
        # Linear transformations
        self.w_qs = nn.Parameter(torch.FloatTensor(n_kernel, d_model, d_trans))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_kernel, d_model, d_trans))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_kernel, d_model, d_trans))
        # Replicated attention
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_kernel * d_trans, n_kernel * d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Initialize weights
        init.xavier_normal_(self.w_qs)
        init.xavier_normal_(self.w_ks)
        init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):
        n_kernel = self.n_kernel
        # residual input
        residual = v
        # In case of different query size
        if (q.size(1) != v.size(1)):
            q = q.repeat(1, v.size(1) // q.size(1), 1)
            k = k.repeat(1, v.size(1) // k.size(1), 1)
        # Obtain different sizes
        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()
        # Collapse the kernels as (kernels X batch X d_model)
        q_s = q.view(n_kernel, mb_size, d_model)
        k_s = k.view(n_kernel, mb_size, d_model)
        v_s = v.view(n_kernel, mb_size, d_model)
        q_s = torch.bmm(q_s, self.w_qs).view(-1, 1, self.d_trans)
        k_s = torch.bmm(k_s, self.w_ks).view(-1, 1, self.d_trans)
        v_s = torch.bmm(v_s, self.w_vs).view(-1, 1, self.d_trans)
        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        if (attn_mask is not None):
            outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_kernel, 1, 1))
        else:
            outputs, attns = self.attention(q_s, k_s, v_s)
        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = outputs.view(mb_size, n_kernel * self.d_trans)#torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)
        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs.view(mb_size, n_kernel, d_model)
        output = self.layer_norm(outputs + residual)
        if (self.average > 0):
            output = torch.mean(output, self.average).unsqueeze(self.average)
        return output, attns

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, d_trans, average=-1, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # Parameters
        self.n_head = n_head
        self.average = average
        self.d_k = d_k
        self.d_v = d_v
        self.d_trans = d_trans
        # First collate layers through conv (reduce dimensionality)
        self.c_q = nn.Conv1d(d_k, d_trans, 13, stride=1, padding=6, bias=True)
        self.c_k = nn.Conv1d(d_k, d_trans, 13, stride=1, padding=6, bias=True)
        self.c_v = nn.Conv1d(d_v, d_trans, 13, stride=1, padding=6, bias=True)
        # Linear transformations
        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_trans))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_trans))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_trans))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_head*d_trans, d_model)
        self.dropout = nn.Dropout(dropout)
        # Now back to original size
        self.final_conv = nn.Conv1d(d_trans, d_v, 13, stride=1, padding=6, bias=True)
        init.xavier_normal_(self.w_qs)
        init.xavier_normal_(self.w_ks)
        init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):
        # Retrieve dimensionalities
        n_head = self.n_head
        # Keep residual
        residual = q
#        print(residual.shape)
        # Collate layers through convolutions
        q_b = self.c_q(q)
        k_b = self.c_k(k)
        v_b = self.c_v(v)
        # Retrieve dimensionalities
        mb_size, len_q, d_model = q_b.size()
        mb_size, len_k, d_model = k_b.size()
        mb_size, len_v, d_model = v_b.size()
#        print(q_b.shape)
#        print(k_b.shape)
#        print(v_b.shape)
        # treat as a (n_head) size batch
        q_s = q_b.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
        k_s = k_b.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
        v_s = v_b.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model
        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, self.d_trans)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, self.d_trans)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, self.d_trans)   # (n_head*mb_size) x len_v x d_v
#        print(q_s.shape)
#        print(k_s.shape)
#        print(v_s.shape)
        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        if (attn_mask is not None):
            outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))
        else:
            outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=None)
        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)
#        print(outputs.shape)
        # project back to residual size
        outputs = self.proj(outputs)
#        print(outputs.shape)
        outputs = self.dropout(outputs)
#        print(d_model)
#        print(outputs.view(-1, self.d_trans, d_model).shape)
        # final convolution (back to original size)
        outputs = self.final_conv(outputs.view(-1, self.d_trans, d_model))
#        print(outputs.shape)
#        print(residual.shape)
        # Add residual and return
        output = self.layer_norm(outputs + residual)
        if (self.average > 0):
            output = torch.mean(output, self.average).unsqueeze(self.average)
        return output, attns

class HierarchicalAttention(torch.nn.Module):
    def __init__(self, nConvs, filters, kernelSize, nFC, hiddenSize, dimEmbedding, nLstm, lstmHidden, dropout, attention, gpu):
        super(HierarchicalAttention, self).__init__()
        self.gpu = gpu
        self.attentionType = attention[0]
        self.attentionAverage = attention[1]
        self.attentionMerge = attention[2]
        self.attentionResidual = attention[3]
        self.attentionQuery = attention[4]
        self.attentionGateLayer = attention[5]
        self.attentionGateOptions = attention[6]
        print('Creating attention model with :')
        print(self.attentionType)
        print(self.attentionAverage)
        print(self.attentionMerge)
        print(self.attentionResidual)
        print(self.attentionQuery)
        print(self.attentionGateLayer)
        print(self.attentionGateOptions)
        # Encoder
        self.nConvs = nConvs
        self.conv, self.bn, self.dropout = [None] * nConvs, [None] * nConvs, [None] * nConvs
        if (attention[0] is not None):
            self.attention_layer = [None] * nConvs
            if (attention[3] is not None):
                self.residual = [None] * nConvs
                self.residual_norm = [None] * nConvs
        # Create convolutions
        for i in range(nConvs):
            if (attention[0] == 'context'):
                self.attention_layer[i] = ScaledDotProductAttention((i == 0 and 1 or filters), dropout, self.attentionAverage)
            if (attention[0] == 'multi'):
                self.attention_layer[i] = KernelAttention((i == 0 and 1 or filters), 128, 8, self.attentionAverage, dropout)
            if (attention[0] == 'repeat'):
                q_size = (i == 0 and 1 or filters)
                if (attention[4] == 'input' or (attention[4] == 'previous' and i == 1) or (attention[4] == 'previous' and attention[1] == 1)):
                    q_size = 1
                self.attention_layer[i] = MultiHeadAttention(16, 128, q_size, (i == 0 and 1 or filters), 32, self.attentionAverage, dropout)
            inSize = (i == 0 and 1 or filters)
            if (attention[3] == 'concat' and i > 0):
                inSize += (i == 1 and 1 or ((self.attentionAverage == 1) and 1 or filters))
            self.conv[i] = nn.Conv1d(inSize, filters, kernelSize, stride=1, padding=int(kernelSize / 2), bias=True)
            self.bn[i] = nn.BatchNorm1d(filters)
            self.dropout[i] = nn.Dropout(dropout)
            if (attention[3] is not 'none'):
                self.residual_norm[i] = LayerNormalization(128)
        self.convs, self.bns, self.dropouts = nn.ModuleList(self.conv), nn.ModuleList(self.bn), nn.ModuleList(self.dropout)
        if (attention[3] is not 'none'):
            self.residual_norms = nn.ModuleList(self.residual_norm)
        if (attention[0] is not None):
            self.attention_layers = nn.ModuleList(self.attention_layer)
            #self.attention_linear = nn.Linear()
            #self.attention_norm = LayerNormalization()
        # Final reduction
        self.f_maxPool = nn.MaxPool1d(2, ceil_mode=True, return_indices=True)
        self.f_batchNorm = nn.BatchNorm1d(filters)
        # Final dimensionality
        self.finalSize = 64 * filters
        if (attention[0] is not None):
            if (attention[2] == 'gate'):
                channelSizes = ((nConvs - 1) * filters) + 1
                self.a_Size = (attention[1] == 1 and (self.nConvs) or channelSizes) * 128
                if (self.attentionGateOptions == 'attend'):
                    self.a_Conv = nn.Conv1d(self.a_Size // 128, self.a_Size // 128, 1, stride=1, padding=0, bias=True)
                    self.a_BN = nn.BatchNorm1d(self.a_Size // 128)
                    self.a_Drop = nn.Dropout(dropout)
                    if (attention[0] == 'context'):
                        self.a_Attend = ScaledDotProductAttention(self.a_Size // 128, dropout, 0)
                    if (attention[0] == 'multi'):
                        self.a_Attend = KernelAttention(self.a_Size // 128, 128, 8, 0, dropout)
                    if (attention[0] == 'repeat'):
                        self.a_Attend = MultiHeadAttention(16, 128, self.a_Size // 128, self.a_Size // 128, 32, 0, dropout)
                if (self.attentionGateLayer > -1):
                    self.gf_Size = (self.attentionGateLayer == (nFC - 1) and dimEmbedding or hiddenSize[self.attentionGateLayer])
                    self.a_Linear = nn.Linear(self.a_Size, self.gf_Size)
                    self.a_Normalize = LayerNormalization(self.gf_Size)
                else:
                    self.a_Linear, self.a_Normalize = [None] * nFC, [None] * nFC
                    for i in range(nFC):
                        gf_Size = (i == (nFC - 1) and dimEmbedding or hiddenSize[i])
                        self.a_Linear[i] = nn.Linear(self.a_Size, gf_Size)
                        self.a_Normalize[i] = LayerNormalization(gf_Size)
                    self.a_Linears, self.a_Normalizes = nn.ModuleList(self.a_Linear), nn.ModuleList(self.a_Normalize)
            # Final hierarchy linear and layer norm
            #self.hierarchy_linear = BottleLinear(self.finalSize, self.finalSize)
            #self.hierarchy_layerNorm = LayerNormalization(self.finalSize, self.finalSize)
        # Fully connected part
        self.nFC = nFC
        self.linear, self.f_bn = [None] * nFC, [None] * nFC
        self.prelu = nn.PReLU()
        for h in range(nFC):
            self.linear[h] = nn.Linear((h == 0 and self.finalSize or hiddenSize[h - 1]), (h == (nFC - 1) and dimEmbedding or hiddenSize[h]))
            self.f_bn[h] = nn.BatchNorm1d((h == (nFC - 1) and dimEmbedding or hiddenSize[h]))
        self.linears, self.f_bns = nn.ModuleList(self.linear), nn.ModuleList(self.f_bn)
        # LSTM part
        self.LSTM = nn.LSTM(dimEmbedding, lstmHidden, nLstm, batch_first=True)
        # Set forget bias to 1
        #for name, p in self.LSTM.named_parameters():
        #    if 'bias' in name:
        #        n = p.size(0)
        #        forget_start_idx, forget_end_idx = n // 4, n // 2
        #        init.constant_(p[forget_start_idx:forget_end_idx], 1)
        self.linear_lstm = nn.Linear(lstmHidden, dimEmbedding)
        # Decoder
        self.linearDec = [None] * nFC
        for h in range(nFC):
            self.linearDec[h] = nn.Linear((h == 0 and dimEmbedding or hiddenSize[len(hiddenSize)-h]), (h == (nFC - 1) and self.finalSize or hiddenSize[len(hiddenSize)-h-1]))
        self.linearsDec = nn.ModuleList(self.linearDec)
        #self.linearTB = nn.Linear(dimEmbedding, self.finalSize)
        self.unpoolTB = nn.MaxUnpool1d(2)
        self.batchnormTB = nn.BatchNorm1d(filters)
        self.convTB = nn.Conv1d(filters, filters, kernel_size=kernelSize, padding=int(kernelSize / 2))
        self.dropoutTB = nn.Dropout(dropout)
        self.bnTB = nn.BatchNorm1d(filters)
        """ Perform reciprocal decoder """
        self.d_conv, self.d_bn, self.d_dropout = [None] * nConvs, [None] * nConvs, [None] * nConvs
        # Create convolutions
        for i in range(nConvs):
            self.d_conv[i] = nn.Conv1d(filters, filters, kernelSize, stride=1, padding=int(kernelSize / 2), bias=True)
            self.d_bn[i] = nn.BatchNorm1d(filters)
            self.d_dropout[i] = nn.Dropout(dropout)
        self.d_convs, self.d_bns, self.d_dropouts = nn.ModuleList(self.d_conv), nn.ModuleList(self.d_bn), nn.ModuleList(self.d_dropout)
        self.deconv1 = nn.ConvTranspose1d(filters, 1, kernelSize, padding=int(kernelSize / 2), stride=1, bias=True)
        # Now initialize all weights
        for m in self.modules():
            #if isinstance(m, nn.Conv1d):
            #    nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
            #elif isinstance(m, nn.BatchNorm1d):
            #    nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x, rho):
        """
        Forward pass of the network.
        Batch size N must be greater than rho (seq length)
        Input : batch of 128-dimensionnal piano-roll vectors (N, 1, 128)
        Output : batch of prediction (N, 1, 128)
        """
        out = x
        self.hierarchy, self.attentions = [None] * (self.nConvs), [None] * (self.nConvs)
        # Encoder convs
        for l in range(self.nConvs):
            if (self.attentionType is not None):
                if (self.attentionQuery == 'layer'):
                    self.hierarchy[l], self.attentions[l] = self.attention_layer[l](out, out, out, None)
                if (self.attentionQuery == 'input'):
                    self.hierarchy[l], self.attentions[l] = self.attention_layer[l](x, x, out, None)
                if (self.attentionQuery == 'previous'):
                    if (l == 0):
                        self.hierarchy[l], self.attentions[l] = self.attention_layer[l](out, out, out, None)
                    else:
                        self.hierarchy[l], self.attentions[l] = self.attention_layer[l](self.hierarchy[l - 1], self.hierarchy[l - 1], out, None)
            if (self.attentionResidual is not None and l > 0):
                if (self.attentionResidual == 'concat'):
                    out = torch.cat((out, self.hierarchy[l - 1]), dim = 1)
                if (self.attentionResidual == 'add'):
                    out = out + self.hierarchy[l - 1]
                    out = self.residual_norm[l - 1](out)
            out = F.relu(self.bn[l](self.conv[l](out)))
            out = self.dropout[l](out)
        # Hierarchical attention (merging part)
        if (self.attentionType is not None):
            hierarchy = torch.cat(self.hierarchy, dim = 1)
            if (self.attentionGateOptions == 'attend'):
                hierarchy = self.a_Drop(self.a_BN(self.a_Conv(hierarchy)))
                hierarchy, _ = self.a_Attend(hierarchy, hierarchy, hierarchy, None)
        """ DIFF 1 = Final conv here """
        out, pool_id = self.f_maxPool(out)
        out = self.f_batchNorm(out)
        out = F.tanh(out)
        # Vectorize output
        out = out.view(-1, self.finalSize)
        # Fully-connected
        for h in range(self.nFC):
            #out = F.tanh((self.linear[h](out)))
            if (h == self.nFC - 1):
                out = F.sigmoid(self.linear[h](out))
            else:
                out = F.relu(self.linear[h](out))
            if (self.attentionGateLayer > -1 and h == self.attentionGateLayer and self.attentionType is not None and self.attentionMerge == 'gate'):
                hierarchy = hierarchy.view(-1, self.a_Size)
                a_out = self.a_Normalize(self.a_Linear(hierarchy))
                out = out * a_out
            if (self.attentionGateLayer == -1 and self.attentionType is not None and self.attentionMerge == 'gate'):
                hierarchy = hierarchy.view(-1, self.a_Size)
                a_out = self.a_Normalize[h](self.a_Linear[h](hierarchy))
                #if (self.attentionGateOptions == 'attend'):
                #    a_out, _ = self.a_Attend[h](a_out, a_out, a_out, None)
                out = out * a_out
        if rho > 0:
            # Decoder part
            decod_in = Variable(torch.Tensor(x.size(0)-rho+1, rho, out.size(1)))
            if self.gpu:
                decod_in = decod_in.cuda()
            for i in range(x.size(0)-rho+1):                                          # take sequence of size rho. From this point N = N - rho
                decod_in[i, :, :] = out[i:i+rho, :]
            # LSTM output
            out_lstm = self.LSTM(decod_in)
            out_lin = self.linear_lstm(out_lstm[0])
            # Select final LSTM code
            out_lin = out_lin[:, -1, :]
            # Inverse fully-connected
            for h in reversed(range(self.nFC)):
                out_lin = F.relu(torch.matmul(out_lin, self.linear[h].weight))
            #for h in range(self.nFC):
            #    if (h == self.nFC - 1):
            #        out_lin = F.sigmoid(self.linearDec[h](out_lin))
            #    else:
            #        out_lin = F.relu(self.linearDec[h](out_lin))
            #out_lin = F.relu(self.linearTB(out_lin))
            # Decoder inverse convs
            out_dec = F.relu(self.batchnormTB(self.unpoolTB(out_lin.view(x.size(0)-rho+1, (self.linear[0].in_features // 64), 64), pool_id[rho-1:,:,:])))
            out_dec = self.convTB(out_dec)
            out_dec = self.dropoutTB(out_dec)
            out_dec = F.relu(self.bnTB(out_dec))
            for l in range(1,self.nConvs):
                if (l == self.nConvs - 1):
                    out_dec = self.d_conv[l](out_dec)
                else:
                    out_dec = F.relu(self.d_bn[l](self.d_conv[l](out_dec)))
                    out_dec = self.d_dropout[l](out_dec)
            out_dec = self.deconv1(out_dec)
            out_dec = F.sigmoid(out_dec)
            return out_dec, out.data
        else:
            return out.data

if __name__ == '__main__':
    nConvs = 4
    filters = 200
    kernelSize = 13
    nFC = 3
    hiddenSize = [1500, 500, 50]
    dimEmbedding = 10
    nLstm = 1
    lstmHidden = 500
    dropout = 0.5
    """
    Attention setup is a quadruplet
    [0] = Type of attention {None, 'context', 'multi', 'repeat'}
    [1] = Dimension of averaging (0 = No averaging / 1 = average features)
    [2] = Type of merging {'concat', 'add', 'condition', 'gate', 'fc'}
    [3] = Residual connexions {None, 'concat', 'add'}
    [4] = Query type {'layer', 'input', 'previous'}
    """
    attention = [None] * 7
    for typ in ['context', 'multi']:
        for average in [1, 0]:
            #for merge in ['gate', 'condition']:
            for merge in ['gate']:
                for residual in ['none']:
                    for query in ['input', 'previous']:
                        for layer in [0]:#[0, 1, 2, -1]:
                            for options in ['none', 'attend']:
                                attention[0] = typ
                                attention[1] = average
                                attention[2] = merge
                                attention[3] = residual
                                attention[4] = query
                                attention[5] = layer
                                attention[6] = options
                                print('Model type :')
                                print(attention)
                                gpu = False
                                testModel = HierarchicalAttention(nConvs, filters, kernelSize, nFC, hiddenSize, dimEmbedding, nLstm, lstmHidden, dropout, attention, gpu)
                                # Create some fake data
                                rho = 12
                                fakeData = torch.zeros(64, 1, 128)
                                out = testModel(Variable(fakeData, requires_grad=True), rho)
                                print('Final output :')
                                print(out[0].shape)
