#!/usr/bin/env python

from typing import List

import numpy
import torch
import torch.nn


class SampleRnnTier(torch.nn.Module):
    def __init__(self, signal_size: int, conditioning_size: int, hidden_size: int, output_size: int, num_layers: int):
        super().__init__()

        self.rnn = torch.nn.GRU(
            input_size=signal_size + conditioning_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.hidden_state = None
        self.output_filter = torch.nn.Linear(hidden_size, output_size)

    def forward(self, signal, conditioning_input=None):
        if conditioning_input is None:
            conditioning_input = torch.empty(0)
        rnn_input = torch.cat([signal, conditioning_input], dim=-1)
        rnn_output, self.hidden_state = self.rnn(rnn_input, self.hidden_state)
        return self.output_filter(rnn_output)


class Encoder(torch.nn.Module):
    def __init__(self, fanins: List[int], encoded_bits: int, hidden_size: int, num_layers: int):
        super().__init__()

        input_sizes = numpy.cumprod([1] + fanins)
        conditioning_sizes = [0] + [fanin * hidden_size for fanin in fanins]
        # The tier without conditioning is first
        self.tiers = torch.nn.ModuleList([
            SampleRnnTier(input_size, conditioning_size, hidden_size, hidden_size, num_layers)
            for input_size, conditioning_size in zip(input_sizes, conditioning_sizes)])
        self.output_filter = torch.nn.Linear(hidden_size, encoded_bits)

    @property
    def frame_len(self):
        pass

    def forward(self, samples):
        """
        samples has shape (batch size × signal size)
        """
        # The data has shape (batch size × num frames × frame size)
        # TODO assert that the signal is a multiple of the frame size
        (batch_size, num_frames) = samples.size()
        samples = samples.unsqueeze(2)

        conditioning = None
        for tier in self.tiers:
            fan_in = 2 # TODO
            if conditioning is not None:
                conditioning = conditioning.view(batch_size, num_frames, -1)
            conditioning = tier(samples.view(batch_size, num_frames, -1), conditioning)
            num_frames /= fan_in

        bits = torch.sign(torch.tanh(self.output_filter(conditioning)))
        return bits


class TrainingDecoder(torch.nn.Module):
    def __init__(self, fanouts: List[int], encoded_bits, hidden_size, num_layers):
        super().__init__()

        input_sizes = reversed(numpy.cumprod([1] + fanouts))
        output_sizes = [fanout * hidden_size for fanout in fanouts] + [1]
        # The tier that takes the encoding as conditioning is first
        self.tiers = torch.nn.ModuleList([
            SampleRnnTier(input_size, hidden_size, hidden_size, output_size, num_layers)
            for input_size, output_size in zip(input_sizes, output_sizes)])
        self.encoding_filter = torch.nn.Linear(encoded_bits, hidden_size)

    def forward(self, samples, encodings):
        (batch_size, signal_size) = samples.size()
        frame_size = 4 #TODO
        num_frames = signal_size / frame_size
        conditioning = self.encoding_filter(encodings)
        for tier in self.tiers:
            samples = samples.view(batch_size, num_frames, -1)
            conditioning = conditioning.view(batch_size, num_frames, -1)
            padded_samples = torch.cat([torch.zeros(batch_size, 1, samples.size()[-1]), samples], dim=1)[:, :-1, :]
            conditioning = tier(padded_samples, conditioning)
            fan_out = 2 # TODO
            num_frames *= fan_out

        return conditioning


class Vocoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        fanins = [2, 2]
        self.encoder = Encoder(fanins, 32, 512, 2)
        self.decoder = TrainingDecoder(fanins, 32, 512, 2)

    def forward(self, samples):
        encoded = self.encoder(samples)
        decoded = self.decoder(samples, encoded)
        return decoded
