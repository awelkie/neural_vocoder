#!/usr/bin/env python

import torch

from vocoder import Vocoder

def main():
    vocoder = Vocoder()
    data = torch.randn(10, 4)
    vocoder(data)

if __name__ == '__main__':
    main()
