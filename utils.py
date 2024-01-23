import torch
import random
import os
import math
import numpy as np
from datetime import datetime
from torch.nn import functional as F
from torch.distributions import Categorical

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# sample action from actor output probability dist
# or get action log_prob if action != None
def sample(logit, action = None):
	dist = Categorical(logits = logit)
	action = dist.sample() if action == None else action
	return action, dist.log_prob(action), dist.entropy()

def rhythm_to_duration(rhythm):
	durations = {   # 4th note = 120
		'00': 480 + 480, '1.': 480 + 240, '41': 120 + 480, '14': 480 + 120, '81': 480 + 60,
		'18': 480 + 60, '01': 480, '2:': 240 + 120 + 60, '2.': 240 + 120, '28': 240 + 60,
		'82': 240 + 60, '02': 240, '4:': 120 + 60 + 30, '4.': 120 + 60, '04': 120,
		'8:': 60 + 30 + 15, '8.': 60 + 30, '08': 60, 'S:': 30 + 15 + 8, 'S.': 30 + 15,
		'16': 30, 'T:': 15 + 8 + 4, 'T.': 15 + 8, '32': 15, '64': 8
	}

	return durations[rhythm]

def duration_to_rhythm(dur):
	rhythms = {
		960: '00', 720: '1.', 600: '14', 540: '18', 480: '01', 420: '2:', 360: '2.', 300: '82',
		240: '02', 210: '4:', 180: '4.', 120: '04', 105: '8:', 90: '8.', 60: '08', 53: 'S:',
		45: 'S.', 30: '16', 27: 'T:', 23: 'T.', 15: '32', 8: '64'
	}

	return rhythms[dur]

def rhythm_to_duration2(rhythm):
	durations = {
		'01': 15, '02': 30, '03': 45, '04': 60, '05': 75, '06': 90, '07': 105, '08': 120, '09': 135, 
		'10': 150, '11': 165, '12': 180, '13': 195, '14': 210, '15': 225, '16': 240, '17': 255, '18': 270, 
		'19': 285, '20': 300, '21': 315, '22': 330, '23': 345, '24': 360, '25': 375, '26': 390, '27': 405, 
		'28': 420, '29': 435, '30': 450, '31': 465, '32': 480
	}

	return durations[rhythm]

def duration_to_rhythm2(dur):
	rhythms = {
		15: '01', 30: '02', 45: '03', 60: '04', 75: '05', 90: '06', 105: '07', 120: '08', 135: '09',
		150: '10', 165: '11', 180: '12', 195: '13', 210: '14', 225: '15', 240: '16', 255: '17', 270: '18',
		285: '19', 300: '20', 315: '21', 330: '22', 345: '23', 360: '24', 375: '25', 390: '26', 405: '27',
		420: '28', 435: '29', 450: '30', 465: '31', 480: '32'
	}
	return rhythms[dur]

def rhythm_to_frame(rhythm):
	# 1 bar = 32 frames, ignore 'S:', 'T.', 'T:' '64'
	frames = {
		'00': 64, '1.': 48, '41': 40, '14': 40, '81': 36, '18': 36, '01': 32, '2:': 28,
		'2.': 24, '28': 20, '82': 20, '02': 16, '4:': 14, '4.': 12, '04': 8, '8:': 7,
		'8.': 6, '08': 4, 'S.': 3, '16': 2, '32': 1
	}

	return frames[rhythm]

def frame_to_rhythm(frame):
	# 1 bar = 32 frames, ignore 'S:', 'T.', 'T:' '64'
	rhythms = {
		64: '00', 48: '1.', 40: '14', 36: '18', 32: '01', 28: '2:', 24: '2.', 20: '82', 16: '02',
		14: '4:', 12: '4.', 8: '04', 7: '8:', 6: '8.', 4: '08', 3: 'S.', 2: '16', 1: '32'
	}

	return rhythms[frame]

def split_song(song, split_ticks = 480):
	bar_ids = []
	dur = 0

	for i in range(len(song) // 4):
		bar_ids.append(dur // split_ticks)
		dur += rhythm_to_duration(song[4 * i + 2:4 * i + 4])

	bars = ['' for i in range(bar_ids[-1] + 1)]

	for i, bar_id in enumerate(bar_ids):
		bars[bar_id] += song[4 * i:4 * i + 4]

	return bars

def split_notes(notes):
	bar_ids = []
	frames = 0

	for note in notes:
		bar_ids.append(frames // 32)
		frames += rhythm_to_frame(note[2:])

	bars = [[] for i in range(bar_ids[-1] + 1)]

	for i, bar_id in enumerate(bar_ids):
		bars[bar_id].append(notes[i])

	return bars

def print_dict(d):
	k_max = 0
	v_max = 0
	for k, v in d.items():
		if len(str(k)) > k_max:
			k_max = len(str(k))
		if len(str(v)) > v_max:
			v_max = len(str(v))

	print('-' * (k_max + v_max + 2))
	for k, v in d.items():
		print(str(k) + ' ' * (k_max - len(str(k))) + ': ' + str(v))
	print('-' * (k_max + v_max + 2))

def pitch_to_midi(pitch):
	names = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
	if pitch == '00':
		return -1
	return int(pitch[1]) * 12 + names.index(pitch[0])

def midi_to_pitch(midi):
	names = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
	if midi == -1:
		return '00'
	return names[midi % 12] + str(midi // 12)

# sample from a list or dict of probabilities
def sample_from(obj):
	if isinstance(obj, list):
		return random.sample(obj, 1)[0]
	elif isinstance(obj, dict):
		keys = list(obj.keys())
		values = [obj[k] for k in keys]
		dist = Categorical(torch.tensor(values, dtype = torch.float))
		return keys[int(dist.sample())]
	else:
		assert False, f'{type(obj)} not implemented'

def now():
	return datetime.now().replace(microsecond = 0)

# backup all .py files
def backup_py(task_id):
	with open(f'backup/backup_{task_id}.txt', 'w', encoding = 'utf-8') as f:
		for name in os.listdir('.'):
			if name.endswith('.py'):
				with open(name, encoding = 'utf-8') as rf:
					f.write(f'####### {name}\n')
					f.write(rf.read() + '\n')

# split string s into a list of n-length substring
# e.g. split_by('080804', 2) == ['08', '08', '04']
def split_by(s, n):
	assert len(s) % n == 0
	return [s[n * i:n * i + n] for i in range(len(s) // n)]