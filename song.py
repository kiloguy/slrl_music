import csv
import math
import numpy as np
from collections import Counter
from utils import *

song_names = []
song_notes = []
# notes = set()
pitches = set()
rhythms = set()
# trans = []
# trans_2 = []
# pitch_trans = []
# pitch_trans_2 = []
# patterns = set()	# rhythm patterns
# patterns_odd, patterns_even = set(), set()
# patterns_odd_probs, patterns_even_probs = Counter(), Counter()
# patterns_240 = set()
# patterns_240_first, patterns_240_second = set(), set()
# avg_bar_notes_count = 0
# pairs = set()
# triples = set()
# hrs = set()
hr_pairs = set()
rh_pairs = set()
# pitch_hrs = set()
# pitch_hrs = []

with open('song2.csv', newline = '', encoding = 'utf-8') as f:
	for row in csv.DictReader(f):
		song_notes.append(split_by(row['Ynote'], 4))
		song_names.append(row['songName'])

# song_notes = song_notes[:100]

for notes in song_notes:
	for note in notes:
		# notes.add(song[4 * i:4 * i + 4])
		pitches.add(note[:2])
		rhythms.add(note[2:])
		# if note[2:] == '81':
		# 	print(note[2:])
		# 	for bar in split_notes(notes):
		# 		print(bar)
		# 	print(''.join(notes))

# notes = sorted(list(notes))
pitches = sorted(list(pitches))
rhythms = sorted(list(rhythms))
# trans = [[0 for j in range(len(notes))] for i in range(len(notes))]
# trans_2 = [[0 for j in range(len(notes))] for i in range(len(notes))]
# pitch_trans = [[0 for j in range(len(pitches))] for i in range(len(pitches))]
# pitch_trans_2 = [[0 for j in range(len(pitches))] for i in range(len(pitches))]

# pattern_pairs = set()

n_pitches = len(pitches)
n_rhythms = len(rhythms)

song_bars = []

for notes in song_notes:
	bars = split_notes(notes)
	song_bars.append(bars)
	# hr = ''
	# pitch_hr = ''

	for b, bar in enumerate(bars):
		# for i in range((len(bar) // 4) - 1):
		# 	trans[notes.index(bar[4 * i:4 * i + 4])][notes.index(bar[4 * i + 4:4 * i + 8])] += 1
		# 	pitch_trans[pitches.index(bar[4 * i:4 * i + 2])][pitches.index(bar[4 * i + 4:4 * i + 6])] += 1
		# 	pairs.add((bar[4 * i:4 * i + 2], bar[4 * i + 4:4 * i + 6]))
		# for i in range((len(bar) // 4) - 2):
		# 	trans_2[notes.index(bar[4 * i:4 * i + 4])][notes.index(bar[4 * i + 8:4 * i + 12])] += 1
		# 	pitch_trans_2[pitches.index(bar[4 * i:4 * i + 2])][pitches.index(bar[4 * i + 8:4 * i + 10])] += 1
		# 	triples.add((bar[4 * i:4 * i + 2], bar[4 * i + 4:4 * i + 6], bar[4 * i + 8:4 * i + 10]))

		# if sum([rhythm_to_frame(bar[4 * i + 2:4 * i + 4]) for i in range(len(bar) // 4)]) == 32:
		# 	pattern = ''.join([bar[4 * i + 2:4 * i + 4] for i in range(len(bar) // 4)])
		# 	patterns.add(pattern)
		# 	if b % 2 == 0:
		# 		patterns_odd.add(pattern)
		# 		patterns_odd_probs[pattern] += 1
		# 		if b + 1 < len(bars) and sum([rhythm_to_duration(bars[b + 1][4 * i + 2:4 * i + 4]) for i in range(len(bars[b + 1]) // 4)]) == 480:
		# 			pattern_pairs.add((pattern, ''.join([bars[b + 1][4 * i + 2:4 * i + 4] for i in range(len(bars[b + 1]) // 4)])))
		# 	else:
		# 		patterns_even.add(pattern)
		# 		patterns_even_probs[pattern] += 1

		# hr += bar[:4] + bar[-4:]
		# pitch_hr += bar[:2] + bar[-4:-2]
		hr_pairs.add((bar[0][:2], bar[-1][:2]))
		if b > 0:
			rh_pairs.add((bars[b - 1][-1][:2], bar[0][:2]))
	# hrs.add(hr)
	# pitch_hrs.add(pitch_hr)
	# pitch_hrs.append(pitch_hr)

# for song in songs:
# 	sections = split_song(song, 240)
# 	for i, section in enumerate(sections):
# 		if sum([rhythm_to_duration(section[4 * i + 2:4 * i + 4]) for i in range(len(section) // 4)]) == 240:
# 			patterns_240.add(''.join([section[4 * i + 2:4 * i + 4] for i in range(len(section) // 4)]))
# 			if i % 2 == 0:
# 				patterns_240_first.add(''.join([section[4 * i + 2:4 * i + 4] for i in range(len(section) // 4)]))
# 			else:
# 				patterns_240_second.add(''.join([section[4 * i + 2:4 * i + 4] for i in range(len(section) // 4)]))

# patterns = sorted(list(patterns))
# patterns_odd = sorted(list(patterns_odd))
# patterns_even = sorted(list(patterns_even))
# patterns_240 = sorted(list(patterns_240))
# patterns_240_first = sorted(list(patterns_240_first))
# patterns_240_second = sorted(list(patterns_240_second))
# pairs = sorted(list(pairs))
# triples = sorted(list(triples))
# hrs = sorted(list(hrs))
# # pitch_hrs = sorted(list(pitch_hrs))
# pattern_pairs = sorted(list(pattern_pairs))
hr_pairs = sorted(list(hr_pairs))
rh_pairs = sorted(list(rh_pairs))

# S = sum(patterns_odd_probs.values())
# patterns_odd_probs = {k: v / S for k, v in dict(patterns_odd_probs).items()}
# S = sum(patterns_even_probs.values())
# patterns_even_probs = {k: v / S for k, v in dict(patterns_even_probs).items()}

# for tran in trans:
# 	total = sum(tran)
# 	if total != 0:
# 		for i in range(len(notes)):
# 			tran[i] /= total

# for tran in trans_2:
# 	total = sum(tran)
# 	if total != 0:
# 		for i in range(len(notes)):
# 			tran[i] /= total

# for tran in pitch_trans:
# 	total = sum(tran)
# 	if total != 0:
# 		for i in range(len(pitches)):
# 			tran[i] /= total

# for tran in pitch_trans_2:
# 	total = sum(tran)
# 	if total != 0:
# 		for i in range(len(pitches)):
# 			tran[i] /= total

# notelens = []

# for song in songs:
# 	for i in range(len(song) // 4):
# 		notelens.append(rhythm_to_duration(song[i * 4 + 2:i * 4 + 4]))

# notelen_avg = float(np.array(notelens).mean())
# notelen_std = float(np.array(notelens).std())

# pitches_dist = [0 for i in range(len(pitches))]

# for song in songs:
# 	for i in range(len(song) // 4):
# 		pitches_dist[pitches.index(song[4 * i:4 * i + 2])] += 1

# sum_pitches_dist = sum(pitches_dist)
# pitches_dist = [v / sum_pitches_dist for v in pitches_dist]

# patterns_odd_onset = []
# patterns_even_onset = []
# patterns_odd_onset_8 = set()
# patterns_even_onset_8 = set()

# for pattern in patterns_odd:
# 	onset = []
# 	for i in range(len(pattern) // 2):
# 		onset.append(1)
# 		onset.extend([0 for j in range(rhythm_to_duration(pattern[2 * i:2 * i + 2]) // 15 - 1)])
# 	patterns_odd_onset.append(onset)

# 	for i in range(24):
# 		patterns_odd_onset_8.add(sum([(2 ** j) * k for j, k in enumerate(onset[i:i + 8])]))

# for pattern in patterns_even:
# 	onset = []
# 	for i in range(len(pattern) // 2):
# 		onset.append(1)
# 		onset.extend([0 for j in range(rhythm_to_duration(pattern[2 * i:2 * i + 2]) // 15 - 1)])
# 	patterns_even_onset.append(onset)

# 	for i in range(24):
# 		patterns_even_onset_8.add(sum([(2 ** j) * k for j, k in enumerate(onset[i:i + 8])]))

# patterns_odd_onset_8 = [[int(o & (2 ** i) != 0) for i in range(8)] for o in list(patterns_odd_onset_8)]
# patterns_even_onset_8 = [[int(o & (2 ** i) != 0) for i in range(8)] for o in list(patterns_even_onset_8)]

# pattern_pairs = set()
# pattern_triples = set()

# for sng in songs:
# 	bars = split_song(sng)
# 	for i in range(len(bars) - 1):
# 		pattern = ''.join([bars[i][4 * j + 2:4 * j + 4] for j in range(len(bars[i]) // 4)])
# 		pattern_next = ''.join([bars[i + 1][4 * j + 2:4 * j + 4] for j in range(len(bars[i + 1]) // 4)])
# 		if sum([rhythm_to_duration(pattern[2 * j:2 * j + 2]) for j in range(len(pattern) // 2)]) == 480 and sum([rhythm_to_duration(pattern_next[2 * j:2 * j + 2]) for j in range(len(pattern_next) // 2)]) == 480:
# 			pattern_pairs.add((pattern, pattern_next))

# 	for i in range(len(bars) - 2):
# 		pattern = ''.join([bars[i][4 * j + 2:4 * j + 4] for j in range(len(bars[i]) // 4)])
# 		pattern_next = ''.join([bars[i + 1][4 * j + 2:4 * j + 4] for j in range(len(bars[i + 1]) // 4)])
# 		pattern_next_next = ''.join([bars[i + 2][4 * j + 2:4 * j + 4] for j in range(len(bars[i + 2]) // 4)])
# 		if sum([rhythm_to_duration(pattern[2 * j:2 * j + 2]) for j in range(len(pattern) // 2)]) == 480 and sum([rhythm_to_duration(pattern_next[2 * j:2 * j + 2]) for j in range(len(pattern_next) // 2)]) == 480 and sum([rhythm_to_duration(pattern_next[2 * j:2 * j + 2]) for j in range(len(pattern_next) // 2)]) == 480:
# 			pattern_triples.add((pattern, pattern_next, pattern_next_next))

# pattern_pairs = sorted(list(pattern_pairs))
# pattern_triples = sorted(list(pattern_triples))(pattern_triples))