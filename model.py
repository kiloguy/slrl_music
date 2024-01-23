import math
import torch
import torch.nn as nn
from utils import *

# for RNN (sl) and 'pi' and 'v' network of PPO
class LSTMNetwork(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dim = 128, n_lstm_layers = 3):
		super(LSTMNetwork, self).__init__()
		self.hidden_dim = hidden_dim
		self.n_lstm_layers = n_lstm_layers
		self.input_dim, self.output_dim = input_dim, output_dim
		self._pe = self.make_pe(10000, 128).to(device)

		self.linear1 = nn.Sequential(nn.Linear(input_dim, 128), nn.Tanh())
		self.linear2 = nn.Sequential(nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
		self.lstm = nn.LSTM(128, self.hidden_dim, self.n_lstm_layers, batch_first = True)
		self.linear3 = nn.Sequential(nn.Linear(self.hidden_dim, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, output_dim))

	# forward shape
	# input: [input_dim] or [batch, input_dim]
	# hidden[0] and hidden[1]: [n_lstm_layers, hidden_dim] or [batch, n_lstm_layers, hidden_dim]
	# output: [output_dim] or [batch, output_dim]
	# hidden_out[0] and hidden_out[1]: [n_lstm_layers, hidden_dim] or [batch, n_lstm_layers, hidden_dim]
	def forward(self, input, hidden = None, pos = None, seq_len = None):
		output = self.linear1(input)
		output = output + self.pe(pos, seq_len)
		output = self.linear2(output)
		output = output.unsqueeze(-2) # insert a seq_len dimension that seq_len = 1

		if hidden != None and len(hidden[0].shape) == 3: # is batched
			hidden = (hidden[0].permute(1, 0, 2).contiguous(), hidden[1].permute(1, 0, 2).contiguous())

		output, hidden_out = self.lstm(output, hidden)

		if len(hidden_out[0].shape) == 3: # is batched
			hidden_out = (hidden_out[0].permute(1, 0, 2), hidden_out[1].permute(1, 0, 2))

		return self.linear3(output).squeeze(-2), hidden_out

	def get_parameters_count(self):
		count = 0
		for param in self.parameters():
			c = 1
			for d in param.shape:
				c *= d
			count += c
		return count

	# reference:
	# paper: "Positional Encoding to Control Output Sequence Length"
	# https://github.com/takase/control-length/blob/master/encdec/fairseq/modules/sinusoidal_positional_embedding.py
	def make_pe(self, seq_len, d):
		emb = [[0 for i in range(d)] for n in range(seq_len)]
		for n in range(seq_len):
			for i in range(d):
				if i % 2 == 0:
					emb[n][i] = math.sin((seq_len - n) / math.pow(10000, i / d))
				else:
					emb[n][i] = math.cos((seq_len - n) / math.pow(10000, (i -  1) / d))
		return torch.tensor(emb, dtype = torch.float)

		# [0,    1,    ..., -seq_len, ..., -seq_len + pos, ..., 9998, 9999]
		# [9999, 9998, ...,                                     1,    0   ]

	def pe(self, pos, seq_len):
		return self._pe[-seq_len + pos]
