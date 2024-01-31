import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from torch.distributions import Categorical
from model import LSTMNetwork
from utils import *

lr_rl         = 0.0002
batch_size_rl = 64
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 5
c1            = 0.5
c2            = 0.01

# reference:
# https://github.com/nikhilbarhate99/PPO-PyTorch
# https://github.com/seungeunrho/minimalRL/blob/master/ppo-lstm.py
class PPOLSTM(nn.Module):
	def __init__(self, state_dim, n_actions):
		super(PPOLSTM, self).__init__()
		self.pi = LSTMNetwork(state_dim, n_actions).to(device)
		self.v = LSTMNetwork(state_dim, 1).to(device)

		self.states = []
		self.hiddens = []
		self.poses = []
		self.seq_lens = []
		self.actions = []
		self.rewards = []
		self.log_probs = []	# old log pi(a|s)
		self.is_dones = []

		self.optimizer = optim.Adam(self.parameters(), lr = lr_rl)
		self.MseLoss = nn.MSELoss()

	def zero_hidden(self):
		return (
			torch.zeros([self.pi.n_lstm_layers, self.pi.hidden_dim], dtype = torch.float).to(device),
			torch.zeros([self.pi.n_lstm_layers, self.pi.hidden_dim], dtype = torch.float).to(device)
		)

	def append_sample(self, state, hidden, pos, seq_len, action, reward, log_prob, is_done):
		self.states.append(state)
		self.hiddens.append((hidden[0].detach(), hidden[1].detach()))
		self.poses.append(pos)
		self.seq_lens.append(seq_len)
		self.actions.append(action)
		self.rewards.append(reward)
		self.log_probs.append(log_prob.detach())
		self.is_dones.append(is_done)

	def act(self, state, hidden = None, pos = None, seq_len = None):
		logit, hidden_next = self.pi(state, hidden, pos, seq_len)
		action, log_prob, _ = sample(logit)
		return action, log_prob, hidden_next

	def update(self, last_state, last_hidden, last_pos, lsat_seq_len):
		# need one more state (and hidden, pos, seq_len) to compute GAE
		self.states.append(last_state)
		self.hiddens.append((last_hidden[0].detach(), last_hidden[1].detach()))
		self.poses.append(last_pos)
		self.seq_lens.append(lsat_seq_len)

		# compute GAE
		# https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
		rewards = []
		gae = 0

		states = torch.stack(self.states)
		hiddens = (
			torch.stack([hidden[0] for hidden in self.hiddens]),
			torch.stack([hidden[1] for hidden in self.hiddens])
		)
		poses = torch.tensor(self.poses, dtype = torch.long)
		seq_lens = torch.tensor(self.seq_lens, dtype = torch.long)

		values, _ = self.v(states, hiddens, poses, seq_lens)
		for i in reversed(range(len(self.rewards))):
			non_terminal = not self.is_dones[i]
			delta = self.rewards[i] + gamma * values[i + 1] * non_terminal - values[i]
			gae = delta + gamma * lmbda * non_terminal * gae
			rewards.insert(0, gae + values[i])

		states = states[:-1]
		hiddens = (hiddens[0][:-1], hiddens[1][:-1])
		poses = poses[:-1]
		seq_lens = seq_lens[:-1]

		# Normalizing the rewards
		rewards = torch.tensor(rewards, dtype = torch.float32).to(device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		# convert list to tensor
		actions = torch.stack(self.actions)
		log_probs = torch.stack(self.log_probs)

		n_samples = len(states)

		# logging for the final epoch
		sum_loss, sum_loss_clip, sum_loss_vf, sum_loss_s, sum_loss_kl = 0, 0, 0, 0, 0
		n_final_update = 0

		for e in range(K_epoch):
			# random permutation for batch update
			# https://github.com/DLR-RM/stable-baselines3/blob/c4f54fcf047d7bf425fb6b88a3c8ed23fe375f9b/stable_baselines3/common/buffers.py       
			indices = torch.randperm(n_samples)
			start_idx = 0

			# create batch from data by permutated indices
			while start_idx < n_samples:
				b_states = states[indices[start_idx:start_idx + batch_size_rl]]
				b_hiddens = (hiddens[0][indices[start_idx:start_idx + batch_size_rl]], hiddens[1][indices[start_idx:start_idx + batch_size_rl]])
				b_poses = poses[indices[start_idx:start_idx + batch_size_rl]]
				b_seq_lens = seq_lens[indices[start_idx:start_idx + batch_size_rl]]
				b_actions = actions[indices[start_idx:start_idx + batch_size_rl]]
				b_log_probs = log_probs[indices[start_idx:start_idx + batch_size_rl]]
				b_rewards = rewards[indices[start_idx:start_idx + batch_size_rl]]

				b_logits, _ = self.pi(b_states, b_hiddens, b_poses, b_seq_lens)
				_, b_new_log_probs, entropy = sample(b_logits, action = b_actions) # new log pi(a|s)
				
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient
				# ratio: (pi_theta / pi_theta_old)
				ratios = torch.exp(b_new_log_probs - b_log_probs)

				# Surrogate Loss
				values, _ = self.v(b_states, b_hiddens, b_poses, b_seq_lens)
				values = values.squeeze().detach()
				advantages = b_rewards - values
				surr1 = ratios * advantages
				surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

				# final loss
				loss_clip, loss_vf, loss_s = -torch.min(surr1, surr2), c1 * self.MseLoss(values, b_rewards), -c2 * entropy
				loss = loss_clip + loss_vf + loss_s

				if e == K_epoch - 1:
					sum_loss += loss.mean().item()
					sum_loss_clip += loss_clip.mean().item()
					sum_loss_vf += loss_vf.mean().item()
					sum_loss_s += loss_s.mean().item()
					n_final_update += 1
				
				self.optimizer.zero_grad()
				loss.mean().backward()
				self.optimizer.step()

				start_idx += batch_size_rl

		self.states, self.hiddens, self.poses, self.seq_lens, self.actions, self.rewards, self.log_probs, self.is_dones = [], [], [], [], [], [], [], []

		return {
			'loss': sum_loss / n_final_update,
			'loss_clip': sum_loss_clip / n_final_update,
			'loss_vf': sum_loss_vf / n_final_update,
			'loss_s': sum_loss_s / n_final_update
		}