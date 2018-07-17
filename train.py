#!/usr/bin/env python

import torch

import vocoder

def main():
    fanins = [2, 2]
    enc = vocoder.Encoder(fanins, 32, 512, 2)
    data = torch.randn(10, 4)
    enc(data)

if __name__ == '__main__':
    main()
