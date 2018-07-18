#!/usr/bin/env python

import torch
import torch.optim
import torch.utils.data
import torchaudio

from vocoder import Vocoder

# Data is 32-bit signed, change to 8 bit unsigned
def quantize(data):
    return ((data >> (32 - 8)) + 128).long()


def main():
    data_dir = 'data'
    data_loader = torch.utils.data.DataLoader(torchaudio.datasets.VCTK(data_dir, download=True, dev_mode=True))

    model = Vocoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for (data, _) in data_loader:
        print('starting batch')
        data = data.squeeze()[:16].unsqueeze(0)
        print(data.size())
        optimizer.zero_grad()
        reconstructed = model(data)
        data_quantized = quantize(data)
        print(reconstructed.size())
        print(data_quantized.size())
        loss = torch.nn.functional.nll_loss(reconstructed.permute(0, 2, 1), data_quantized)
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
