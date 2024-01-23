import torch
import torch.nn as nn
from PPOLSTM import PPOLSTM
from utils import *
from envs import *

# @torch.no_grad()
# def generate_rhythms(model_r, model_b, n_bars = 8):
# 	env = EnvR()
# 	profiles, info_b = generate_profiles(model_b, n_bars)
# 	state, pos = env.reset(n_bars = n_bars, profiles = profiles)
# 	hidden = model_r.zero_hidden()
	
# 	ep_reward_r = 0
# 	seq_len = 32 * env.n_bars

# 	while True:
# 		action, _, hidden_next = model_r.act(state, hidden, pos, seq_len)
# 		state_next, pos, _, reward, done, info = env.step(action.cpu().numpy())
# 		ep_reward_r += reward

# 		if done:
# 			info['ep_reward_r'] = ep_reward_r
# 			info = {**info, **info_b}
# 			return env.rhythms, info

# 		state = state_next
# 		hidden = hidden_next

@torch.no_grad()
def generate_profiles(model_b, n_bars = 8):
	env = EnvB()
	state, pos = env.reset(n_bars = n_bars)
	hidden = model_b.zero_hidden()
	
	ep_reward_b = 0
	seq_len = 32 * env.n_bars

	while True:
		action, _, hidden_next = model_b.act(state, hidden, pos, 32 * env.n_bars)
		state_next, pos, _, reward, done, info = env.step(action.cpu().numpy())
		ep_reward_b += reward

		if done:
			info['ep_reward_b'] = ep_reward_b
			return env.profiles, info

		state = state_next
		hidden = hidden_next

def train_rl_pr(n_steps, trained_p = None, trained_r = None, trained_b = None):
	env_p, env_r = EnvP(), EnvR()
	model_p = PPOLSTM(env_p.state_dim, env_p.n_actions) if trained_p == None else trained_p
	model_r = PPOLSTM(env_r.state_dim, env_r.n_actions) if trained_r == None else trained_r
	print(model_p)
	print(model_r)

	ep_infos = {}
	ep = 0
	step = 0

	while step < n_steps:
		# n_bars = sample_from([8, 16, 24, 32])
		n_bars = 8

		profiles = generate_profiles(trained_b, n_bars = n_bars)[0]
		state_p, pos = env_p.reset(n_bars = n_bars, profiles = profiles)
		state_r, pos = env_r.reset(n_bars = n_bars, profiles = profiles, init_notes = env_p.init_notes)
		hidden_p, hidden_r = model_p.zero_hidden(), model_r.zero_hidden()
		
		ep_reward_p, ep_reward_r = 0, 0
		seq_len = 32 * n_bars

		while step < n_steps:
			action_p, log_prob_p, hidden_next_p = model_p.act(state_p, hidden_p, pos, seq_len)
			action_r, log_prob_r, hidden_next_r = model_r.act(state_r, hidden_r, pos, seq_len)
			state_next_p, pos, is_init, reward_p, done, info_p = env_p.step(action_p.cpu().numpy(), action_r.cpu().numpy())
			state_next_r, pos, is_init, reward_r, done, info_r = env_r.step(action_p.cpu().numpy(), action_r.cpu().numpy())
			if done or not is_init:
				model_p.append_sample(state_p, hidden_p, pos, seq_len, action_p, reward_p, log_prob_p, done)
				model_r.append_sample(state_r, hidden_r, pos, seq_len, action_r, reward_r, log_prob_r, done)
				step += 1
			ep_reward_p += reward_p
			ep_reward_r += reward_r

			if done:
				info_p['epreward_p'] = ep_reward_p
				info_r['epreward_r'] = ep_reward_r
				info = {**info_p, **info_r}
				print(info)
				for k, v in info.items():
					ep_infos[k] = v if not k in ep_infos else ep_infos[k] + v
				ep += 1
				break

			state_p = state_next_p
			state_r = state_next_r
			hidden_p = hidden_next_p
			hidden_r = hidden_next_r

	ep_infos = {k: v / ep for k, v in ep_infos.items()}
	log_dict_p = model_p.update(state_next_p, hidden_next_p, pos, seq_len)
	log_dict_r = model_r.update(state_next_r, hidden_next_r, pos, seq_len)
	log_dict_p = {k + '_p': v for k, v in log_dict_p.items()}
	log_dict_r = {k + '_r': v for k, v in log_dict_r.items()}
	log_dict = {**log_dict_p, **log_dict_r, **ep_infos}
	print_dict({k: round(v, 4) for k, v in log_dict.items()})

	return model_p, model_r, log_dict

def train_rl_b(n_steps, trained):
	env = EnvB()
	model = PPOLSTM(env.state_dim, env.n_actions) if trained == None else trained
	print(model)

	ep_infos = {}
	ep = 0
	step = 0

	while step < n_steps:
		# n_bars = sample_from([8, 16, 24, 32])
		n_bars = 8

		state, pos = env.reset(n_bars = n_bars)
		hidden = model.zero_hidden()
		
		ep_reward = 0
		seq_len = 32 * n_bars

		while step < n_steps:
			action, log_prob, hidden_next = model.act(state, hidden, pos, seq_len)
			state_next, pos, is_init, reward, done, info = env.step(action.cpu().numpy())
			if done or not is_init:
				model.append_sample(state, hidden, pos, seq_len, action, reward, log_prob, done)
				step += 1
			ep_reward += reward

			if done:
				info[f'ep_reward_b'] = ep_reward
				print(info)
				for k, v in info.items():
					ep_infos[k] = v if not k in ep_infos else ep_infos[k] + v
				ep += 1
				break

			state = state_next
			hidden = hidden_next

	ep_infos = {k: v / ep for k, v in ep_infos.items()}
	log_dict = model.update(state_next, hidden_next, pos, seq_len)
	log_dict = {k + '_b': v for k, v in log_dict.items()}
	log_dict = {**log_dict, **ep_infos}
	print_dict({k: round(v, 4) for k, v in log_dict.items()})

	return model, log_dict

# @torch.no_grad()
def test_rl(n_episodes, model_p, model_r, model_b, n_bests = 0):
	print(model_p)
	print(model_r)
	print(model_b)
	env_p = EnvP()
	env_r = EnvR()

	ep_infos = {}
	bests = []

	for ep in range(n_episodes):
		print('ep', ep)
		n_bars = 8
		seq_len = n_bars * 32

		profiles, info_b = generate_profiles(model_b, n_bars = n_bars)
		state_p, pos = env_p.reset(n_bars = n_bars, profiles = profiles)
		state_r, pos = env_r.reset(n_bars = n_bars, profiles = profiles, init_notes = env_p.init_notes)
		hidden_p, hidden_r = model_p.zero_hidden(), model_r.zero_hidden()
		
		ep_reward_p, ep_reward_r = 0, 0

		while True:
			action_p, _, hidden_next_p = model_p.act(state_p, hidden_p, pos, seq_len)
			action_r, _, hidden_next_r = model_r.act(state_r, hidden_r, pos, seq_len)
			state_next_p, pos, is_init, reward_p, done, info_p = env_p.step(action_p.cpu().numpy(), action_r.cpu().numpy())
			state_next_r, pos, is_init, reward_r, done, info_r = env_r.step(action_p.cpu().numpy(), action_r.cpu().numpy())
			ep_reward_p += reward_p
			ep_reward_r += reward_r

			if done:
				info = {**info_p, 'ep_reward_p': ep_reward_p, **info_r, 'ep_reward_r': ep_reward_r, **info_b}
				print(info)
				
				bests.append({**info, 'bars': env_p.bars, 'profiles': env_p.profiles})
				bests = sorted(bests, reverse = True, key = lambda best: best['ep_reward_p'] + best['ep_reward_r'] + best['ep_reward_b'])[:n_bests]

				for k, v in info.items():
					ep_infos[k] = v if not k in ep_infos else ep_infos[k] + v
				break

			state_p = state_next_p
			state_r = state_next_r
			hidden_p = hidden_next_p
			hidden_r = hidden_next_r

	avg_ep_info = {k: v / n_episodes for k, v in ep_infos.items()}
	print_dict({k: round(v, 4) for k, v in avg_ep_info.items()})
	return avg_ep_info, bests

if __name__ == '__main__':
	## test
	model_p = torch.load('trained/pr_0515_4.0M_p.pth', map_location = device)
	model_r = torch.load('trained/pr_0515_4.0M_r.pth', map_location = device)
	model_b = torch.load('trained/pr_0515_4.0M_b.pth', map_location = device)
	# model_p.pi = torch.load('trained/sl_pr_0520_62k_p.pth', map_location = device)
	# model_r.pi = torch.load('trained/sl_pr_0520_62k_r.pth', map_location = device)
	# model_b.pi = torch.load('trained/sl_pr_0520_62k_b.pth', map_location = device)
	avg_ep_info, bests = test_rl(
		10000,
		model_p,
		model_r,
		model_b,
		# n_bests = 10
	)

	for best in bests:
		print(best['profiles'])
		for notes in best['bars']:
			print(sum([rhythm_to_frame(note[2:]) for note in notes]), ''.join(notes))
		print(''.join([''.join(notes) for notes in best['bars']]))
		del best['bars']
		del best['profiles']
		print(best)
	print(avg_ep_info)
	##