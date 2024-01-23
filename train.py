import sys
import csv
import song
import torch
import torch.nn as nn
from PPOLSTM import PPOLSTM
from utils import *
from envs import *
from sl import train_sl
from rl import train_rl_b, train_rl_pr

total_rl_steps  = 4000000
rl_update_steps = 4096
sl_update_steps = 64

task_id = sys.argv[1] if len(sys.argv) >= 2 else now().strftime('%y%m%d_%H%M%S')
trained_id = sys.argv[2] if len(sys.argv) >= 3 else None
backup_py(task_id)
log_f = open(f'logs/log_{task_id}.csv', 'w', newline = '')
csv_writer = None

model_p = torch.load(f'trained/{trained_id}_p.pth', map_location = device) if trained_id != None else None
model_r = torch.load(f'trained/{trained_id}_r.pth', map_location = device) if trained_id != None else None
model_b = torch.load(f'trained/{trained_id}_b.pth', map_location = device) if trained_id != None else None
rl_step = 0
sl_step = 0
start_time = now()

while rl_step < total_rl_steps:
	model_b, info_b = train_rl_b(n_steps = rl_update_steps, trained = model_b)
	model_p, model_r, info_pr = train_rl_pr(n_steps = rl_update_steps, trained_p = model_p, trained_r = model_r, trained_b = model_b)
	rl_step += rl_update_steps
	info = {'timestep': rl_step, **info_pr, **info_b}
	info = {k: round(v, 4) for k, v in info.items()}
	print_dict(info)

	if csv_writer == None:
		csv_writer = csv.DictWriter(log_f, fieldnames = list(info.keys()))
		csv_writer.writeheader()

	csv_writer.writerow(info)
	log_f.flush()

	model_b.pi = train_sl('b', n_steps = sl_update_steps, trained = model_b.pi, start_time = start_time, start_step = sl_step)
	model_p.pi = train_sl('p', n_steps = sl_update_steps, trained = model_p.pi, start_time = start_time, start_step = sl_step)
	model_r.pi = train_sl('r', n_steps = sl_update_steps, trained = model_r.pi, start_time = start_time, start_step = sl_step)
	sl_step += sl_update_steps

	torch.save(model_p, f'trained/{task_id}_p.pth')
	torch.save(model_r, f'trained/{task_id}_r.pth')
	torch.save(model_b, f'trained/{task_id}_b.pth')

	if rl_step // 100000 != (rl_step - rl_update_steps) // 100000:
		torch.save(model_p, f'trained/{task_id}_{rl_step // 100000 / 10}M_p.pth')
		torch.save(model_r, f'trained/{task_id}_{rl_step // 100000 / 10}M_r.pth')
		torch.save(model_b, f'trained/{task_id}_{rl_step // 100000 / 10}M_b.pth')

log_f.close()
print('-' * 40)
print(f'task_id: {task_id}')
print(f'training time: {now() - start_time}')
print('-' * 40)