import numpy as np
import random
import song
from utils import *
from kmodes import *

state_dim_p = song.n_pitches + song.n_rhythms + 4 + 8 + 1 + 1 + n_profiles
# state_dim_p = song.n_pitches + song.n_rhythms + n_profiles
n_actions_p = song.n_pitches
state_dim_r = song.n_pitches + song.n_rhythms + 4 + 8 + 1 + 1 + n_profiles
# state_dim_r = song.n_pitches + song.n_rhythms + n_profiles
n_actions_r = song.n_rhythms
state_dim_b = n_profiles + 1 + 1
# state_dim_b = n_profiles
n_actions_b = n_profiles

def random_init_notes():
	rhythms = [note[2:] for note in split_notes(sample_from(song.song_notes))[0]]
	pitches = [note[:2] for note in sample_from(song.song_notes)[:len(rhythms)]]
	return [pitches[i] + rhythms[i] for i in range(len(rhythms))]

class EnvP:
	def __init__(self):
		self.state_dim = song.n_pitches + song.n_rhythms + 4 + 8 + 1 + 1 + n_profiles
		# self.state_dim = song.n_pitches + song.n_rhythms + n_profiles
		self.n_actions = song.n_pitches

	def step(self, action_p, action_r):
		is_init = len(self.notes) < len(self.init_notes)
		if is_init:
			self.cur_bar.append(self.init_notes[len(self.notes)])
		else:
			self.cur_bar.append(song.pitches[action_p] + song.rhythms[action_r])

		self.notes.append(self.cur_bar[-1])
		cur_frames = rhythm_to_frame(self.notes[-1][2:])
		self.frames += rhythm_to_frame(self.notes[-2][2:])
		reward = 0

		##### when generated a note
		if not self.notes[-1][0] in 'CDEGA':
			reward -= 1
			self.n_nopenta += 1
		jump = abs(pitch_to_midi(self.notes[-1][:2]) - pitch_to_midi(self.notes[-2][:2]))
		if len(self.cur_bar) > 1:
			self.n_j += 1
			if jump > 12:
				reward -= 1
				self.n_lj += 1
		else:
			self.n_bj += 1
			if jump > 14:
				reward -= 1
				self.n_lbj += 1

			if not (self.bars[-1][-1][:2], self.cur_bar[0][:2]) in song.rh_pairs:
				self.n_bad_rh += 1
				reward -= 1

		if jump == 5:
			reward -= 2.5
			self.n_4th += 1

		# if len(self.bars) >= 2:
		# 	for note_p in self.bars[-1]:
		# 		note = self.notes[-1]
		# 		if note[:2] == note_p[:2] and (rhythm_to_frame(note[2:]) >= 8 and rhythm_to_frame(note_p[2:]) <= 4 or rhythm_to_frame(note[2:]) <= 4 and rhythm_to_frame(note_p[2:]) >= 8):
		# 			self.n_voice_accom += 1
		# 			reward += 1.5
		# 			break

		if len(self.notes) > 2:
			pre_jump = abs(pitch_to_midi(self.notes[-2][:2]) - pitch_to_midi(self.notes[-3][:2]))
			if (pre_jump == 3 or pre_jump == 4) and (jump == 3 or jump == 4):
				reward -= 2.5
				self.n_repeat_3rd += 1

		if self.frames // 32 != (self.frames + cur_frames) // 32:
		##### when generated a bar
			for i in range(len(self.cur_bar) - 2):
				j1 = pitch_to_midi(self.cur_bar[i + 1][:2]) - pitch_to_midi(self.cur_bar[i][:2])
				j2 = pitch_to_midi(self.cur_bar[i + 2][:2]) - pitch_to_midi(self.cur_bar[i + 1][:2])
				if abs(j1) == 2 and abs(j2) == 3 or abs(j1) == 3 and abs(j2) == 2:
					self.n_bars_has_j23 += 1
					reward += 1
					break

			if not (self.cur_bar[0][:2], self.cur_bar[-1][:2]) in song.hr_pairs:
				self.n_bad_hr += 1
				reward -= 1

			print(sum([rhythm_to_frame(note[2:]) for note in self.cur_bar]), ''.join(self.cur_bar))
			self.bars.append(self.cur_bar)
			self.cur_bar = []

		done = self.frames + cur_frames >= 32 * self.n_bars

		if done:
		##### when done
			end_in_cga = True
			if not self.notes[-1][0] in ['C', 'G', 'A']:
				reward -= self.n_bars * 1.5
				end_in_cga = False

			print(''.join(self.notes))

		return self.to_state(), self.frames, is_init, reward, done, {
			'n_bars_has_j23': self.n_bars_has_j23,
			'large_jump': self.n_lj / self.n_j, 'large_bar_jump': self.n_lbj / self.n_bj,
			'n_bad_hr': self.n_bad_hr, 'n_bad_rh': self.n_bad_rh,
			'end_in_cga': int(end_in_cga),
			'n_nopenta': self.n_nopenta,
			'n_repeat_3rd': self.n_repeat_3rd,
			'n_4th': self.n_4th
			# 'n_voice_accom': self.n_voice_accom
		} if done else {}

	def reset(self, profiles, n_bars = 8, init_notes = None):
		self.profiles, self.n_bars = profiles.copy(), n_bars
		self.init_notes = init_notes.copy() if init_notes != None else random_init_notes()
		print(f'n_bars: {self.n_bars}, profiles: {self.profiles}, init_melody: {"".join(self.init_notes)}')
		self.profiles.extend([0, 0])

		self.notes = [self.init_notes[0]]
		self.cur_bar = [self.init_notes[0]]
		self.bars = []
		self.frames = 0

		self.n_bars_has_j23 = 0
		self.n_j, self.n_lj, self.n_bj, self.n_lbj = 0, 0, 0, 0
		self.n_bad_hr = 0
		self.n_bad_rh = 0
		self.n_nopenta = 0
		self.n_repeat_3rd = 0
		self.n_4th = 0
		self.n_voice_accom = 0
		return self.to_state(), self.frames

	def to_state(self):
		r = torch.zeros(self.state_dim, dtype = torch.float)
		r[song.pitches.index(self.notes[-1][:2])] = 1
		r[song.n_pitches + song.rhythms.index(self.notes[-1][2:])] = 1
		r[song.n_pitches + song.n_rhythms + ((self.frames % 32) // 8)] = 1
		r[song.n_pitches + song.n_rhythms + 4 + ((self.frames % 32) % 8)] = 1
		r[song.n_pitches + song.n_rhythms + 4 + 8] = (self.frames // 32) % 2
		r[song.n_pitches + song.n_rhythms + 4 + 8 + 1] = self.frames / (32 * self.n_bars)
		r[song.n_pitches + song.n_rhythms + 4 + 8 + 1 + 1 + self.profiles[(self.frames + rhythm_to_frame(self.notes[-1][2:])) // 32]] = 1
		# r[song.n_pitches + song.n_rhythms + self.profiles[(self.frames + rhythm_to_frame(self.notes[-1][2:])) // 32]] = 1
		return r.to(device)

class EnvR:
	def __init__(self):
		self.state_dim = song.n_pitches + song.n_rhythms + 4 + 8 + 1 + 1 + n_profiles
		# self.state_dim = song.n_pitches + song.n_rhythms + n_profiles
		self.n_actions = song.n_rhythms

	def step(self, action_p, action_r):
		is_init = len(self.notes) < len(self.init_notes)
		if is_init:
			self.cur_bar.append(self.init_notes[len(self.notes)])
		else:
			self.cur_bar.append(song.pitches[action_p] + song.rhythms[action_r])

		self.notes.append(self.cur_bar[-1])
		cur_frames = rhythm_to_frame(self.notes[-1][2:])
		self.frames += rhythm_to_frame(self.notes[-2][2:])
		reward = 0

		##### when generated a note
		if rhythm_to_frame(self.notes[-1][2:]) > 8:
			# reward -= 4
			self.n_over_04 += 1
		elif len(self.notes) >= 3 and self.notes[-1][2:] == '04' and self.notes[-2][2:] == '04' and self.notes[-3][2:] == '04':
			# reward -= 4
			self.n_repeat_3_04 += 1
		if not self.notes[-1][0] in 'CDEGA' and rhythm_to_frame(self.notes[-1][2:]) > 4:
			reward -= 2.5
			self.n_long_nopenta += 1

		if len(self.notes) >= 3 and rhythm_to_frame(self.notes[-1][2:]) >= 16 and rhythm_to_frame(self.notes[-2][2:]) >= 16 and rhythm_to_frame(self.notes[-3][2:]) >= 16:
			reward -= 6
			self.n_3_long += 1

		if self.frames // 32 != (self.frames + cur_frames) // 32:
		##### when generated a bar
			if len(self.bars) % 2 == 1:
				if sum([rhythm_to_frame(note[2:]) for note in self.bars[-1]]) + sum([rhythm_to_frame(note[2:]) for note in self.cur_bar]) != 64:
					self.n_misalign += 1
					reward -= 2.5

			pattern = ''.join([note[2:] for note in self.cur_bar])
			if len(self.patterns) >= 4 and pattern == self.patterns[-1] and pattern == self.patterns[-2] and pattern == self.patterns[-3] and pattern == self.patterns[-4]:
				reward -= 10
				self.n_repeat_5_pattern += 1
			elif len(self.patterns) >= 1 and pattern == self.patterns[-1]:
				reward += 2.5
				self.n_repeat_prev_pattern += 1

			self.patterns.append(pattern)
			self.bars.append(self.cur_bar)
			self.cur_bar = []

		done = self.frames + cur_frames >= 32 * self.n_bars

		if done:
		##### when done
			short_last_note = int(rhythm_to_frame(self.notes[-1][2:]) < 16)
			reward -= short_last_note * self.n_bars * 2.5

			onsets = []
			target_onsets = []
			nearest_profile_dist = 0

			for note in self.notes:
				onsets.append(1)
				onsets.extend([0 for i in range(rhythm_to_frame(note[2:]) - 1)])

			for i in range(self.n_bars):
				target_onsets.extend(profile_onsets[self.profiles[i]])
				nearest_profile_dist += min([kmodes_distance(onsets[32 * i:32 * i + 32], profile_onsets[k], 32) for k in range(n_profiles)])

		return self.to_state(), self.frames, is_init, reward, done, {
			'short_last_note': short_last_note,
			'n_misalign': self.n_misalign,
			'profile_dist': kmodes_distance(onsets[:32 * self.n_bars], target_onsets, 32 * self.n_bars),
			'nearest_profile_dist': nearest_profile_dist,
			'n_repeat_prev_pattern': self.n_repeat_prev_pattern,
			'n_repeat_5_pattern': self.n_repeat_5_pattern,
			# 'n_over_04': self.n_over_04,
			# 'n_repeat_3_04': self.n_repeat_3_04,
			'n_long_nopenta': self.n_long_nopenta,
			'n_3_long': self.n_3_long
		} if done else {}

	def reset(self, profiles, n_bars = 8, init_notes = None):
		self.profiles, self.n_bars = profiles.copy(), n_bars
		self.init_notes = init_notes.copy() if init_notes != None else random_init_notes()
		self.profiles.extend([0, 0])

		self.notes = [self.init_notes[0]]
		self.cur_bar = [self.init_notes[0]]
		self.bars = []
		self.frames = 0

		self.patterns = []
		self.n_misalign = 0
		self.n_repeat_prev_pattern = 0
		self.n_repeat_5_pattern = 0
		self.n_over_04 = 0
		self.n_repeat_3_04 = 0
		self.n_long_nopenta = 0
		self.n_3_long = 0
		return self.to_state(), self.frames

	def to_state(self):
		r = torch.zeros(self.state_dim, dtype = torch.float)
		r[song.pitches.index(self.notes[-1][:2])] = 1
		r[song.n_pitches + song.rhythms.index(self.notes[-1][2:])] = 1
		r[song.n_pitches + song.n_rhythms + ((self.frames % 32) // 8)] = 1
		r[song.n_pitches + song.n_rhythms + 4 + ((self.frames % 32) % 8)] = 1
		r[song.n_pitches + song.n_rhythms + 4 + 8] = (self.frames // 32) % 2
		r[song.n_pitches + song.n_rhythms + 4 + 8 + 1] = self.frames / (32 * self.n_bars)
		r[song.n_pitches + song.n_rhythms + 4 + 8 + 1 + 1 + self.profiles[(self.frames + rhythm_to_frame(self.notes[-1][2:])) // 32]] = 1
		# r[song.n_pitches + song.n_rhythms + self.profiles[(self.frames + rhythm_to_frame(self.notes[-1][2:])) // 32]] = 1
		return r.to(device)

class EnvB:
	def __init__(self):
		self.state_dim = n_profiles + 1 + 1
		# self.state_dim = n_profiles
		self.n_actions = n_profiles

	def step(self, action):
		is_init = len(self.profiles) < len(self.init_profiles)
		self.profiles.append(self.init_profiles[len(self.profiles)] if is_init else int(action))
		done = len(self.profiles) >= self.n_bars
		reward = 0

		if len(self.profiles) >= 5 and self.profiles[-1] == self.profiles[-2] and self.profiles[-2] == self.profiles[-3] and self.profiles[-3] == self.profiles[-4] and self.profiles[-4] == self.profiles[-5]:
			reward -= 10
			self.n_repeat_5_profile += 1
		elif self.profiles[-1] == self.profiles[-2]:
			reward += 2.5
			self.n_repeat_prev_profile += 1

		if done:
			short_last_note_profile = int(rhythm_to_frame(profile_patterns[self.profiles[-1]][-2:]) < 16)
			reward -= short_last_note_profile * self.n_bars * 2
			print(self.profiles)

		return self.to_state(), 32 * (len(self.profiles) - 1), is_init, reward, done, {
			'short_last_note_profile': short_last_note_profile,
			'n_repeat_5_profile': self.n_repeat_5_profile,
			'n_repeat_prev_profile': self.n_repeat_prev_profile
		} if done else {}

	def reset(self, n_bars = 8, init_profiles = None):
		self.n_bars = n_bars
		self.init_profiles = init_profiles.copy() if init_profiles != None else [random.randint(0, n_profiles - 1)]
		print(f'n_bars: {self.n_bars}, init_profiles: {self.init_profiles}')
		self.profiles = [self.init_profiles[0]]

		self.n_repeat_5_profile = 0
		self.n_repeat_prev_profile = 0
		return self.to_state(), 32 * (len(self.profiles) - 1)

	def to_state(self):
		r = torch.zeros(self.state_dim, dtype = torch.float)
		r[self.profiles[-1]] = 1
		r[n_profiles] = (len(self.profiles) - 1) % 2
		r[n_profiles + 1] = (len(self.profiles) - 1) / self.n_bars
		return r.to(device)