"""
Implements GPT model. The bulk of the code here is based on Andrej Karpathy's minGPT implementation.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.0
    resid_pdrop = 0.0
    attn_pdrop = 0.0

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT_alef(GPTConfig):
    """ Roughly 110M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class GPT_bet(GPTConfig):
    """ Roughly 336M params """
    n_layer = 24
    n_head = 16
    n_embd = 1024


class GPT_gimel(GPTConfig):
    """ Roughly 730M params """
    n_layer = 36
    n_head = 20
    n_embd = 1280


class GPT_dalet(GPTConfig):
    """ Roughly 1.5B params """
    n_layer = 48
    n_head = 25
    n_embd = 1600


class MeanLayer(torch.nn.Module):
    def __init__(self, dim, keepdim=False):
        super(MeanLayer, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        out = torch.mean(x, self.dim, self.keepdim)
        return out


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, config):
        super().__init__()

        # model config
        self.model_config = config

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        print('Number of parameters:', sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."    

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        unreduced_loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1))
            unreduced_loss = F.cross_entropy(logits.permute(0, 2, 1), targets, reduce=False)
            unreduced_loss = unreduced_loss.mean(-1)  # average over pixels

        return logits, loss, unreduced_loss

    def sample(self, x, steps, temperature=1.0, sample=False, top_k=None):
        """
        conditioned on x (shape: b x t), predict the next token, feeding the predictions back into the model each time.
        """
        def top_k_logits(logits, k):
            v, _ = torch.topk(logits, k)
            out = logits.clone()
            out[out < v[:, [-1]]] = -float('Inf')
            return out
    
        block_size = self.get_block_size()
        
        for k in range(steps):
            if k % 100 == 0:
                print('Step {} of {}'.format(k, steps))

            x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
            logits, _, _ = self.forward(x_cond)
            
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution or choose the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)

        return x

    def sample_freely(self, n_samples=1):
        # uniformly sample the first pixel
        counts = torch.ones(self.model_config.vocab_size)
        prob = counts / counts.sum()

        start_pixel = np.random.choice(np.arange(self.model_config.vocab_size), size=(n_samples, 1), replace=True, p=prob.numpy())
        start_pixel = torch.from_numpy(start_pixel)
        if torch.cuda.is_available():
            start_pixel = start_pixel.cuda()

        print('Started unconditional sampling.')    
        pixels = self.sample(start_pixel, self.get_block_size(), temperature=1.0, sample=True, top_k=128)

        return pixels

    def sample_from_half(self, x, n_samples=2):
        print('Started conditional sampling.')
        all_pixels = []
        ctx_len = (self.get_block_size() + 1) // 2

        all_pixels.append(x)  # append the original images first
        for i in range(n_samples-1):
            print('Sample {} of {}'.format(i, n_samples-1))
            pixels = self.sample(x[:, :ctx_len], ctx_len, temperature=1.0, sample=True, top_k=128)
            all_pixels.append(pixels)

        return torch.cat(all_pixels)


class LinearProbeGPT(nn.Module):
    """ Optional: GPT with a linear classifier head attached """
    def __init__(self, tok_emb, pos_emb, drop, blocks, ln_1, head):
        super().__init__()

        # input embedding stem
        self.tok_emb = tok_emb
        self.pos_emb = pos_emb
        self.drop = drop
        self.blocks = blocks
        self.ln_1 = ln_1
        self.head = head

        print('Number of parameters:', sum(p.numel() for p in self.parameters()))

    def forward(self, idx):
        b, t = idx.size()

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_1(x)
        x = torch.mean(x, 1, False)
        logits = self.head(x)

        return logits