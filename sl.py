import song
import torch
from kmodes import *
from utils import *
from model import LSTMNetwork
from envs import *

lr_sl         = 0.0005
batch_size_sl = 64

### prepare training data
seqs_pr = []
seqs_b = []
inputs_pr = []
inputs_b = []
targets_pr = []
targets_b = []
class_indices_p = [[] for i in range(song.n_pitches)]
class_indices_r = [[] for i in range(song.n_rhythms)]
class_indices_b = [[] for i in range(n_profiles)]

for notes in song.song_notes:
	seq_pr = []
	seq_b = []
	onsets = []
	profiles = []

	for note in notes:
		onsets.append(1)
		onsets.extend([0 for i in range(rhythm_to_frame(note[2:]) - 1)])

	if len(onsets) % 32 != 0:
		onsets.append(1)
		onsets.extend([0 for i in range(32 - (len(onsets) % 32))])

	for i in range(len(onsets) // 32):
		pattern = onsets[32 * i:32 * i + 32]
		dists = [kmodes_distance(pattern, profile_onsets[k], 32) for k in range(n_profiles)]
		profiles.append(dists.index(min(dists)))
		seq_b.append((profiles[-1], i % 2, i * 32 / len(onsets), i * 32, len(onsets)))

	profiles.append(-1)
	frames = 0
	total_frames = sum([rhythm_to_frame(note[2:]) for note in notes])
	for note in notes:
		seq_pr.append((
			song.pitches.index(note[:2]), song.rhythms.index(note[2:]),
			(frames % 32) // 8, (frames % 32) % 8, (frames // 32) % 2,
			frames / total_frames, profiles[(frames + rhythm_to_frame(note[2:])) // 32], frames, total_frames
		))
		frames += rhythm_to_frame(note[2:])

	seqs_pr.append(seq_pr)
	seqs_b.append(seq_b)

for seq_pr in seqs_pr:
	for i in range(len(seq_pr)):
		if sum([rhythm_to_frame(song.rhythms[note[1]]) for note in seq_pr[:i]]) < 32:
			continue
		inputs_pr.append(seq_pr[:i])
		targets_pr.append(seq_pr[i][:2])
		class_indices_p[seq_pr[i][0]].append(len(inputs_pr) - 1)
		class_indices_r[seq_pr[i][1]].append(len(inputs_pr) - 1)

for seq_b in seqs_b:
	for i in range(1, len(seq_b)):
		inputs_b.append(seq_b[:i])
		targets_b.append(seq_b[i][0])
		class_indices_b[seq_b[i][0]].append(len(inputs_b) - 1)
###

def train_sl(mode, n_steps, trained = None, start_time = None, start_step = 0):
	class_indices = {'p': class_indices_p, 'r': class_indices_r, 'b': class_indices_b}[mode]
	input_dim = {'p': state_dim_p, 'r': state_dim_r, 'b': state_dim_b}[mode]
	output_dim = {'p': n_actions_p, 'r': n_actions_r, 'b': n_actions_b}[mode]
	inputs = inputs_b if mode == 'b' else inputs_pr
	targets = targets_b if mode == 'b' else targets_pr

	model = LSTMNetwork(input_dim, output_dim).to(device) if trained == None else trained
	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = lr_sl)
	start_time = now() if start_time == None else start_time

	print(model)

	for step in range(n_steps):
		# sample a batch of indices in training data for balance
		indices = []
		while len(indices) < batch_size_sl:
			classes = random.sample(range(output_dim), min(output_dim, batch_size_sl - len(indices)))
			classes = list(filter(lambda clas: len(class_indices[clas]) != 0, classes))
			indices.extend([random.sample(class_indices[clas], 1)[0] for clas in classes])

		indices = list(sorted(indices, key = lambda index: len(inputs[index])))
		n_finish = 0

		max_input_len = max([len(inputs[index]) for index in indices])
		b_inputs = torch.zeros(max_input_len, batch_size_sl, input_dim)
		b_targets = torch.zeros(batch_size_sl, output_dim)
		b_outputs = torch.zeros(batch_size_sl, output_dim)
		b_pos = torch.zeros(max_input_len, batch_size_sl, dtype = torch.long)
		b_len = torch.ones(max_input_len, batch_size_sl, dtype = torch.long)

		# create batched input (all sequences padded to max_input_len)
		for i, index in enumerate(indices):
			if mode == 'b':
				b_targets[i][targets[index]] = 1
			else:
				b_targets[i][targets[index][0 if mode == 'p' else 1]] = 1

			for t, note in enumerate(inputs[index]):
				profile = note
				if mode == 'b':
					b_inputs[t][i][profile[0]] = 1
					b_inputs[t][i][n_profiles] = profile[1]
					b_inputs[t][i][n_profiles + 1] = profile[2]
				else:
					b_inputs[t][i][note[0]] = 1
					b_inputs[t][i][song.n_pitches + note[1]] = 1
					b_inputs[t][i][song.n_pitches + song.n_rhythms + note[2]] = 1
					b_inputs[t][i][song.n_pitches + song.n_rhythms + 4 + note[3]] = 1
					b_inputs[t][i][song.n_pitches + song.n_rhythms + 4 + 8] = note[4]
					b_inputs[t][i][song.n_pitches + song.n_rhythms + 4 + 8 + 1] = note[5]
					b_inputs[t][i][song.n_pitches + song.n_rhythms + 4 + 8 + 1 + 1 + note[6]] = 1

				b_pos[t][i] = note[-2]
				b_len[t][i] = note[-1]

		hidden = None

		b_inputs = b_inputs.to(device)

		# iterate
		for t, b_notes in enumerate(b_inputs):
			output, hidden = model(b_notes, hidden, b_pos[t], b_len[t])

			while n_finish < batch_size_sl and t == len(inputs[indices[n_finish]]) - 1:
				# encounter the end of input sequence n_finish, store into b_outputs[n_finish]
				b_outputs[n_finish] = output[n_finish]
				n_finish += 1

			# only care the last output, so the output during iteration can be detached
			hidden = (hidden[0].detach(), hidden[1].detach())
			output = output.detach()

		loss = loss_fn(b_outputs, b_targets)
		acc = sum(b_outputs.argmax(dim = 1) == b_targets.argmax(dim = 1)).item() / batch_size_sl
		print(f'{now() - start_time} step: {start_step + step}, loss: {loss.item():.4f}, acc: {acc}')

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return model

@torch.no_grad()
def test_sl(mode, trained):
	input_dim = {'p': state_dim_p, 'r': state_dim_r, 'b': state_dim_b}[mode]
	output_dim = {'p': n_actions_p, 'r': n_actions_r, 'b': n_actions_b}[mode]
	seqs = seqs_b if mode == 'b' else seqs_pr
	model = trained
	n_corrects, N = 0, 0

	print(model)

	for seq in seqs:
		hidden = None
		frames = 0

		for t in range(len(seq) - 1):
			note = profile = seq[t]
			inp = torch.zeros(input_dim)

			if mode == 'b':
				inp[profile[0]] = 1
				inp[n_profiles] = profile[1]
				inp[n_profiles + 1] = profile[2]
			else:
				inp[note[0]] = 1
				inp[song.n_pitches + note[1]] = 1
				inp[song.n_pitches + song.n_rhythms + note[2]] = 1
				inp[song.n_pitches + song.n_rhythms + 4 + note[3]] = 1
				inp[song.n_pitches + song.n_rhythms + 4 + 8] = note[4]
				inp[song.n_pitches + song.n_rhythms + 4 + 8 + 1] = note[5]
				inp[song.n_pitches + song.n_rhythms + 4 + 8 + 1 + 1 + note[6]] = 1

			inp = inp.to(device)

			output, hidden = model(inp, hidden, note[-2], note[-1])
			frames += rhythm_to_frame(song.rhythms[note[1]]) if mode != 'b' else 32

			if frames >= 32:
				correct = output.argmax().item() == seq[t + 1][1 if mode == 'r' else 0]
				print(f'{N} correct: {correct}')
				N += 1
				n_corrects += int(correct)

	print(f'acc: {n_corrects / N:.4f} ({n_corrects}/{N})')
	return n_corrects / N

if __name__ == '__main__':
	# start_time = now()
	# model = None

	# for i in range(65):
	# 	model = train_sl('r', 1000, trained = model, start_time = start_time, start_step = i * 1000)
	# 	torch.save(model, f'trained/sl_0502_{i + 1}k_r.pth')

	# start_time = now()
	# model = None

	# for i in range(65):
	# 	model = train_sl('p', 1000, trained = model, start_time = start_time, start_step = i * 1000)
	# 	torch.save(model, f'trained/sl_0502_{i + 1}k_p.pth')

	# start_time = now()
	# model = None

	# for i in range(65):
	# 	model = train_sl('b', 1000, trained = model, start_time = start_time, start_step = i * 1000)
	# 	torch.save(model, f'trained/sl_0502_{i + 1}k_b.pth')

	# test_sl('p', torch.load('trained/sl_pr_0520_62k_p.pth', map_location = device)) # 17483
	# test_sl('r', torch.load('trained/sl_pr_0520_62k_r.pth', map_location = device))
	test_sl('b', torch.load('trained/sl_pr_0520_62k_b.pth', map_location = device)) # 2345