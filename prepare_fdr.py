"""
Dataset for FDR (false discovery rate) prediction
- Correct predicted formula
- Incorrect predicted formula 
"""

import os
import pickle
import argparse
from tqdm import tqdm
import yaml
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset import MS2FDataset
from model_tcn import MS2FNet_tcn
from utils import ATOMS_INDEX_re, formula_to_dict, formula_refinement



def convert_to_list_of_dicts(data_dict):
    keys = list(data_dict.keys())
    num_items = len(data_dict[keys[0]])
    result = []

    for i in range(num_items):
        item_dict = {}
        for key in keys:
            item_dict[key] = data_dict[key][i]
        result.append(item_dict)

    return result

def vec2formula(vec, withH=True):
	formula = ''
	for idx, v in enumerate(vec):
		v = round(float(v))
		
		if v <= 0:
			continue
		elif not withH and ATOMS_INDEX_re[idx] == 'H': 
			continue
		elif v == 1:
			formula += ATOMS_INDEX_re[idx]
		else:
			formula += ATOMS_INDEX_re[idx] + str(v)
	return formula

def eval_step(model, loader, device): 
	model.eval()
	y_true = []
	y_pred = []
	spec_ids = []
	mae = []
	mass_true = []
	mass_pred = []
	mass_mae = []
	with tqdm(total=len(loader)) as bar: 
		for _, batch in enumerate(loader): 
			spec_id, y, x, mass, env = batch
			x = x.to(device).to(torch.float32)
			y = y.to(device).to(torch.float32)
			env = env.to(device).to(torch.float32)
			mass = mass.to(device).to(torch.float32)

			with torch.no_grad(): 
				_, pred_f, pred_mass, _, _ = model(x, env)
				
			bar.set_description('Eval')
			bar.update(1)

			y_true.append(y.detach().cpu())
			y_pred.append(pred_f.detach().cpu())

			mae = mae + torch.mean(torch.abs(y - pred_f), dim=1).tolist()
			spec_ids = spec_ids + list(spec_id)

			mass_true.append(mass.detach().cpu())
			mass_pred.append(pred_mass.detach().cpu())
			mass_mae = mass_mae + torch.abs(mass - pred_mass).tolist()

	y_true = torch.cat(y_true, dim = 0)
	y_pred = torch.cat(y_pred, dim = 0)

	mass_true = torch.cat(mass_true, dim = 0)
	mass_pred = torch.cat(mass_pred, dim = 0)
	return spec_ids, y_true, y_pred, mae, mass_true, mass_pred, mass_mae

def init_random_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	return



if __name__ == "__main__":
	# Training settings
	parser = argparse.ArgumentParser(description='Preprocess the data for FDR prediction')
	parser.add_argument('--train_data', type=str, required=True,
						help='Path to data (.pkl)')
	parser.add_argument('--test_data', type=str, required=True,
						help='Path to data (.pkl)')
	parser.add_argument('--config_path', type=str, required=True,
						help='Path to configuration (.yaml)')
	
	parser.add_argument('--resume_path', type=str, required=True,
						help='Path to pretrained model')
	parser.add_argument('--fdr_dir', type=str, required=True,
						help='Path to save FDR dataset')
	
	parser.add_argument('--seed', type=int, default=42,
						help='Seed for random functions')
	parser.add_argument('--device', type=int, nargs='+', default=[0], 
						help='Which GPUs to use if any (default: [0]). Accepts multiple values separated by space.')
	parser.add_argument('--no_cuda', action='store_true', 
						help='Disables CUDA')
	args = parser.parse_args()
	
	init_random_seed(args.seed)
	start_time = time.time()

	with open(args.config_path, 'r') as f: 
		config = yaml.load(f, Loader=yaml.FullLoader)
	print('Load the model & training configuration from {}'.format(args.config_path))

	device_1st = torch.device("cuda:" + str(args.device[0])) if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
	print(f'Device(s): {args.device}')

	# 1. Model
	model = MS2FNet_tcn(config['model']).to(device_1st)
	num_params = sum(p.numel() for p in model.parameters())
	# print(f'{str(model)} #Params: {num_params}')
	print(f'# MS2FNet_tcn Params: {num_params}')

	if len(args.device) > 1: # Wrap the model with nn.DataParallel
		model = nn.DataParallel(model, device_ids=args.device)
	# need to do something when using one GPU

	print('Loading the best model...')
	model.load_state_dict(torch.load(args.resume_path, map_location=device_1st)['model_state_dict'])

	for data_path, out_path in zip([args.test_data, args.train_data], 
									[args.test_data.replace('_test.pkl', '_fdr_test.pkl'), 
									args.train_data.replace('_train.pkl', '_fdr_train.pkl')]): 
		
		# 2. Data
		dataset = MS2FDataset(data_path)

		# If the dataset is larger than 10000, randomly sampling
		if len(dataset) > 10000: 
			print('Sample 10000 samples for FDR prediction')
			indices = np.arange(len(dataset))
			np.random.shuffle(indices)
			sample_indices = indices[:10000]
			
			sampler = SubsetRandomSampler(sample_indices) # Create a SubsetRandomSampler
			loader = DataLoader(dataset,
								batch_size=config['train']['batch_size'],
								sampler=sampler,  # Use the sampler instead of shuffle
								num_workers=config['train']['num_workers'],
								drop_last=True)
		else:
			loader = DataLoader(dataset,
								batch_size=config['train']['batch_size'], 
								shuffle=True, 
								num_workers=config['train']['num_workers'], 
								drop_last=True)

		# 3. Prediction
		spec_ids, y_true, y_pred, mae, mass_true, mass_pred, mass_mae = eval_step(model, loader, device_1st)
		
		# calculate the formula string, which will be used in postprocessing
		formula_pred = [vec2formula(y) for y in y_pred] 
		formula_true = [vec2formula(y) for y in y_true] 

		# 4. Post-processing
		formula_redined = {'Refined Formula ({})'.format(str(k)): [] for k in range(config['post_processing']['top_k'])}
		# Please note that here we use the experimental precursor m/z, rather than the theoretic precursor m/z. 
		for pred_f, m in tqdm(zip(formula_pred, mass_true), total=len(mass_true), desc='Post'):

			# Use true monoisotopic mass (not experimental precursor m/z) to calculate molmass
			# Extend refine_atom_type with any atoms present in the predicted formula
			# so the search space matches what run_fiddle.py uses at inference time.
			refine_atom_type = list(config['post_processing']['refine_atom_type'])
			refine_atom_num  = list(config['post_processing']['refine_atom_num'])
			for atom, cnt in formula_to_dict(pred_f).items():
				if atom == 'H' or atom in refine_atom_type:
					continue
				refine_atom_type.append(atom)
				refine_atom_num.append(max(1, int(cnt)))

			refined_results = formula_refinement([pred_f], m.item(),
												config['post_processing']['mass_tolerance'],
												config['post_processing']['ppm_mode'],
												config['post_processing']['top_k'],
												config['post_processing']['maxium_miss_atom_num'],
												config['post_processing']['time_out'],
												refine_atom_type,
												refine_atom_num,
												)

			for i, (refined_f, refined_m) in enumerate(zip(refined_results['formula'], refined_results['mass'])): 
				formula_redined['Refined Formula ({})'.format(str(i))].append(refined_f)

		# 5. Check the correctness of refined results and label them for FDR prediction
		info_dict = {'ID': spec_ids, 'Formula': formula_true}
		res_df = pd.DataFrame({**info_dict, **formula_redined})

		# map title with spec & env
		with open(data_path, 'rb') as file: 
			data = pickle.load(file)
			pkl_data = {}
			for d in data:
				pkl_data[d['title']] = [d['spec'], d['env']]
			
		data = {'title': [], 'pred_formula': [], 'spec': [], 'env': [], 'label': []}
		for k in range(config['post_processing']['top_k']): 
			res_df['Label ({})'.format(str(k))] = res_df.apply(lambda x: formula_to_dict(x['Formula']) == \
										formula_to_dict(x['Refined Formula ({})'.format(str(k))]), axis=1)

			correct_df = res_df[res_df['Label ({})'.format(str(k))] == True]
			correct_df = correct_df.dropna(subset=['Refined Formula ({})'.format(str(k))])
			titles = correct_df['ID'].tolist()
			data['title'].extend(titles)
			data['pred_formula'].extend(correct_df['Refined Formula ({})'.format(str(k))].tolist())
			data['label'].extend([1.]*len(correct_df))
			for title in titles:
				spec, env = pkl_data[title]
				data['spec'].append(spec)
				data['env'].append(env)
			print(k, 'correct', len(titles))

			incorrect_df = res_df[res_df['Label ({})'.format(str(k))] == False]
			incorrect_df = incorrect_df.dropna(subset=['Refined Formula ({})'.format(str(k))])
			titles = incorrect_df['ID'].tolist()
			data['title'].extend(titles)
			data['pred_formula'].extend(incorrect_df['Refined Formula ({})'.format(str(k))].tolist())
			data['label'].extend([0.]*len(incorrect_df))
			for title in titles:
				spec, env = pkl_data[title]
				data['spec'].append(spec)
				data['env'].append(env)
			print(k, 'incorrect', len(titles))

		print('\nSave the FDR dataset...')
		# out_path = os.path.join(args.fdr_dir, out_path)
		with open(out_path, 'wb') as f: 
			data = convert_to_list_of_dicts(data) 
			pickle.dump(data, f)
			print('Save {} FDR data to {}'.format(len(data), out_path))
		print('Done!')
		