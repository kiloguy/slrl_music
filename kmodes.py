import random
from collections import Counter
from utils import *
import song

n_profiles = 16 # K
profile_onsets = [
	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
	[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
	[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
	[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
	[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
	[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
]

profile_patterns = [
	'0202',
	'08161616161616080408',
	'4.0808080808',
	'16161616161616161616161604',
	'0808080808080808',
	'0808080802',
	'1616161616160802',
	'04048.160808',
	'08080808040808',
	'0816161616161602',
	'8.161616161608161616161616',
	'8.16080802',
	'08161604081616081616',
	'08080808080804',
	'081616080808161604',
	'8.16080808080808'
]

# profile_onsets = [
# 	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
# 	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
# 	[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
# 	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
# 	[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
# 	[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
# 	[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
# 	[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
# ]

# profile_patterns = [
# 	'0202',
# 	'08080816160816160808',
# 	'0808080808080808',
# 	'081616081616040808',
# 	'08080808080804',
# 	'161616161616161602',
# 	'0808080802',
# 	'028.160808',
# 	'081616080808161604',
# 	'081616080808161616161616',
# 	'161616160816161616161604',
# 	'0816161616161602',
# 	'081616080816160408',
# 	'04080802',
# 	'8.16080808080808',
# 	'161616160408080808'
# ]

def kmodes_distance(x, y, n_features):
	return sum([x[j] != y[j] for j in range(n_features)])

# convert a list of rhythms to nearest list of bar profiles
# input duration will be padded to exactly multiples of 32
def rhythms_to_profiles(rhythms):
	onsets = []
	profiles = []
	for rhythm in rhythms:
		onsets.append(1)
		onsets.extend([0 for j in range(rhythm_to_frame(rhythm) - 1)])
	if len(onsets) % 32 != 0:
		onsets.append(1)
		onsets.extend([0 for i in range(32 - (len(onsets) % 32))])
	for i in range(len(onsets) // 32):
		dists = [kmodes_distance(onsets[32 * i:32 * i + 32], profile_onsets[k], 32) for k in range(n_profiles)]
		profiles.append(dists.index(min(dists)))
	return profiles

def kmodes(data, K, n_features):
	## initialize K center
	# reference: https://github.com/nicodv/kmodes/blob/master/kmodes/util/init_methods.py#L6
	centers = [[] for i in range(K)]

	for j in range(n_features):
		choices = [x[j] for x in data]
		for center in centers:
			center.append(sample_from(choices))

	for i in range(K):
		idx_dists = [(idx, kmodes_distance(centers[i], x, n_features)) for idx, x in enumerate(data)]
		idx_dists = sorted(idx_dists, key = lambda idx_dist: idx_dist[1])
		while len(idx_dists) > 1 and data[idx_dists[0][0]] in centers:
			idx_dists.pop(0)
		centers[i] = data[idx_dists[0][0]]
	##

	old_x_centers, dists = None, None

	while True:
		x_centers = []
		center_xs = [[] for i in range(K)]
		dists = []

		for idx, x in enumerate(data):
			min_dist = kmodes_distance(x, centers[0], n_features)
			min_dist_center = 0

			for i in range(K):
				dist = kmodes_distance(x, centers[i], n_features)
				if dist < min_dist:
					min_dist = dist
					min_dist_center = i

			dists.append(min_dist)
			x_centers.append(min_dist_center)
			center_xs[min_dist_center].append(idx)

		for i in range(K):
			new_center = []
			for j in range(n_features):
				mode = Counter([data[idx][j] for idx in center_xs[i]]).most_common()[0][0]
				new_center.append(mode)
			centers[i] = new_center
		
		if x_centers == old_x_centers:
			break

		old_x_centers = x_centers

	return centers, old_x_centers, dists

if __name__ == '__main__':
	random.seed(0)
	patterns = []

	for notes in song.song_notes:
		onsets = []

		for note in notes:
			onsets.append(1)
			onsets.extend([0 for i in range(rhythm_to_frame(note[2:]) - 1)])

		if len(onsets) % 32 != 0:
			onsets.append(1)
			onsets.extend([0 for i in range(32 - (len(onsets) % 32))])

		for i in range(len(onsets) // 32):
			pattern = onsets[32 * i:32 * i + 32]
			patterns.append(pattern)

	# try 100 times K-modes, find the profiles set with smallest distance to patterns in dataset
	avg_min_dist = 100
	profile_onsets = []
	profile_patterns = []

	for t in range(100):
		centers, _, dists = kmodes(patterns, n_profiles, 32)
		print(t, sum(dists) / len(dists))
		if sum(dists) / len(dists) < avg_min_dist:
			avg_min_dist = sum(dists) / len(dists)
			profile_onsets = centers

	for onsets in profile_onsets:
		pattern = ''
		frames = 1
		for i in range(1, 32):
			if onsets[i] == 1:
				pattern += frame_to_rhythm(frames)
				frames = 1
			else:
				frames += 1

		pattern += frame_to_rhythm(frames)
		print(onsets, pattern)

	print(f'avg_min_dist: {avg_min_dist}')

# profile_onsets_f = [
# 	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
# 	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
# 	[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
# 	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
# 	[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
# 	[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
# 	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
# ]

# profile_patterns_f = [
# 	'0808080802',
# 	'0808080808080808',
# 	'0202',
# 	'01',
# 	'161616161616161602',
# 	'8.16080808080808',
# 	'0816161616161602',
# 	'08080808080804',
# 	'8.16080802',
# 	'08080808040808',
# 	'0208080808',
# 	'16161616161616161616161616161616',
# 	'04080802',
# 	'040808040808',
# 	'080808161602',
# 	'04080808080808'
# ]