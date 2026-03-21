import numpy as np
import re
from tqdm import tqdm

from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from .mol_utils import ATOMS_WEIGHT, monoisotopic_mass_calculator

MSLEVEL_MAP = {
	'MS2': 2,
	'MS1': 1, 
	'2': 2,
	'1': 1,
}
# this is the superset of spectra keys in our data
mgf_key_order = ['title', 'precursor_type', 'precursor_mz', 
				'molmass', 'ms_level', 'ionmode', 
				'source_instrument', 'instrument_type', 'collision_energy', 
				'smiles', 'inchi_key']

def sdf2mgf(path, prefix): 
	'''mgf format
	[{
		'params': {
			'title': prefix_<index>, 
			'precursor_type': <precursor_type (e.g. [M+NH4]+ and [M+H]+)>, 
			'precursor_mz': <precursor m/z>,
			'molmass': <isotopic mass>, 
			'ms_level': <ms_level>, 
			'ionmode': <POSITIVE|NEGATIVE>, 
			'source_instrument': <source_instrument>,
			'instrument_type': <instrument_type>, 
			'collision_energy': <collision energe>, 
			'smiles': <smiles>, 
			'inchi_key': <inchi_key>, 
		},
		'm/z array': mz_array,
		'intensity array': intensity_array
	}, ...]
	'''
	supp = Chem.SDMolSupplier(path)
	print('Read {} data from {}'.format(len(supp), path))

	spectra = []
	for idx, mol in enumerate(tqdm(supp)): 
		if mol == None or \
			not mol.HasProp('MASS SPECTRAL PEAKS') or \
			not mol.HasProp('PRECURSOR TYPE') or \
			not mol.HasProp('PRECURSOR M/Z') or \
			not mol.HasProp('SPECTRUM TYPE') or \
			not mol.HasProp('COLLISION ENERGY') or \
			not mol.HasProp('ION MODE'): 
			continue
		
		mz_array = []
		intensity_array = []
		raw_ms = mol.GetProp('MASS SPECTRAL PEAKS').split('\n')
		for line in raw_ms:
			mz_array.append(float(line.split()[0]))
			intensity_array.append(float(line.split()[1]))
		mz_array = np.array(mz_array)
		intensity_array = np.array(intensity_array)

		inchi_key = 'Unknown' if not mol.HasProp('INCHIKEY') else mol.GetProp('INCHIKEY')
		instrument = 'Unknown' if not mol.HasProp('INSTRUMENT') else mol.GetProp('INSTRUMENT')
		spectrum = {
			'params': {
				'title': prefix+'_'+str(idx), 
				'precursor_type': mol.GetProp('PRECURSOR TYPE'),
				'precursor_mz': mol.GetProp('PRECURSOR M/Z'),
				'molmass': mol.GetProp('EXACT MASS'),
				'ms_level': mol.GetProp('SPECTRUM TYPE'), 
				'ionmode': mol.GetProp('ION MODE'), 
				'source_instrument': instrument,
				'instrument_type': mol.GetProp('INSTRUMENT TYPE'), 
				'collision_energy': mol.GetProp('COLLISION ENERGY'), 
				'smiles': Chem.MolToSmiles(mol, isomericSmiles=True), 
				'inchi_key': inchi_key, 
			},
			'm/z array': mz_array,
			'intensity array': intensity_array
		} 
		spectra.append(spectrum)
	return spectra

def filter_spec(spectra, config, type2charge):
	"""Filter and clean a list of spectra according to a configuration dict.

	Applies sequential filters: instrument type/name, MS level, atom count/type,
	precursor type, peak count, m/z range, and ppm mass error.

	Args:
		spectra: List of spectra dicts in MGF format.
		config: Dict of filter thresholds (keys: 'instrument_type', 'ms_level',
		        'atom_type', 'precursor_type', 'min_peak_num', 'min_mz', 'max_mz',
		        'ppm_tolerance', etc.).
		type2charge: Dict mapping precursor type string to charge int.

	Returns:
		tuple: (clean_spectra, smiles_list) — filtered spectra and their SMILES.
	"""
	clean_spectra = []
	smiles_list = []

	records = {'instrument_type': [], 'instrument': [], 'ms_level': [], 
			'atom_type': [], 'precursor_type': [], 'peak_num': [],
			'max_mz': [], 'invalud_formula': [], 'ppm_tolerance': [], 
			'invalid_intensity_array': []}
	for idx, spectrum in enumerate(tqdm(spectra)): 
		smiles = spectrum['params']['smiles'] 
		mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
		if mol == None: continue

		# Unify the smiles format
		smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
		spectrum['params']['smiles'] = smiles

		# Filter by instrument type
		if 'instrument_type' in config.keys(): 
			instrument_type = spectrum['params']['instrument_type']
			if instrument_type not in config['instrument_type']: 
				records['instrument_type'].append(spectrum['params']['title'])
				continue

		# Filter by instrument (MoNA contains too many intrument names to filter out)
		if 'instrument' in config.keys(): 
			instrument = spectrum['params']['source_instrument']
			if instrument not in config['instrument']: 
				records['instrument'].append(spectrum['params']['title'])
				continue

		# Filter by mslevel
		if 'ms_level' in config.keys(): 
			mslevel = spectrum['params']['ms_level']
			if mslevel != config['ms_level']: 
				records['ms_level'].append(spectrum['params']['title'])
				continue

		# Filter by atom number and atom type 
		if len(mol.GetAtoms()) > config['max_atom_num'] or len(mol.GetAtoms()) < config['min_atom_num']: continue
		is_compound_countain_rare_atom = False 
		for atom in mol.GetAtoms(): 
			if atom.GetSymbol() not in config['atom_type']:
				is_compound_countain_rare_atom = True
				break
		if is_compound_countain_rare_atom: 
			records['atom_type'].append(spectrum['params']['title'])
			continue

		# Filter by precursor type
		precursor_type = spectrum['params']['precursor_type']
		if precursor_type not in config['precursor_type']:
			records['precursor_type'].append(spectrum['params']['title'])
			continue

		# Filt by peak number
		if len(spectrum['m/z array']) < config['min_peak_num']: 
			records['peak_num'].append(spectrum['params']['title'])
			continue

		# Filter by max m/z
		if np.max(spectrum['m/z array']) < config['min_mz'] or np.max(spectrum['m/z array']) > config['max_mz']: 
			records['max_mz'].append(spectrum['params']['title'])
			continue

		# Filter by ppm (mass error)
		if 'ppm_tolerance' in config.keys(): 
			f = CalcMolFormula(mol)
			# try: 
			# 	f = Formula(f)
			# except: # invalud formula
			# 	records['invalud_formula'].append(spectrum['params']['title'])
			# 	continue
			molmass = monoisotopic_mass_calculator(f, 'f')
			theo_mz = precursor_mz_calculator(precursor_type, molmass)
			ppm = abs(theo_mz - float(spectrum['params']['precursor_mz'])) / theo_mz * 10**6
			
			if ppm > config['ppm_tolerance']: 
				records['ppm_tolerance'].append(spectrum['params']['title'])
				continue
			
		# Remove the invalid spectra
		if spectrum['intensity array'].max() == spectrum['intensity array'].min(): 
			records['invalid_intensity_array'].append(spectrum['params']['title'])
			continue

		spectrum['params']['theoretical_precursor_mz'] = theo_mz
		spectrum['params']['ppm'] = ppm
		spectrum['params']['simulated_precursor_mz'] = simulate_experimental_mz(theo_mz, config['ppm_tolerance'])
		spectrum['params']['charge'] = type2charge[precursor_type] # add charge
		clean_spectra.append(spectrum)
		smiles_list.append(smiles)

	print('Filtering records: ')
	for k, v in records.items():
		print('\t{}: {}'.format(k, len(v)))

	return clean_spectra, smiles_list

def simulate_experimental_mz(theoretical_mz, relative_mass_tolerance_ppm):
	"""
	Simulate experimental precursor m/z value by shifting the theoretical value 
	within a Gaussian distribution of mass deviations.

	:param theoretical_mz: Theoretical precursor m/z value.
	:param relative_mass_tolerance_ppm: Relative mass tolerance in parts per million (ppm).
	:return: Simulated experimental precursor m/z value.
	"""
	
	# Calculate the standard deviation of the Gaussian distribution as 1/3 of the relative mass tolerance
	std_dev = (relative_mass_tolerance_ppm / 1e6 * theoretical_mz) / 3
	
	# Randomly sample a mass deviation from the Gaussian distribution
	mass_deviation = np.random.normal(0, std_dev)
	
	# Shift the theoretical m/z value by the sampled mass deviation to simulate the experimental value
	experimental_mz = theoretical_mz + mass_deviation
	
	return experimental_mz

def ce2nce(ce, precursor_mz, charge):
	"""Convert absolute collision energy (eV) to normalized collision energy (NCE).

	Args:
		ce: Collision energy in eV.
		precursor_mz: Precursor m/z value.
		charge: Precursor charge state (int, 1–8).

	Returns:
		float: Normalized collision energy (dimensionless).
	"""
	charge_factor = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}
	return ce * 500 * charge_factor[charge] / precursor_mz

def precursor_mz_calculator(precursor_type, mass):
	"""Compute the expected precursor m/z from neutral monoisotopic mass.

	Args:
		precursor_type: Adduct string, e.g. '[M+H]+'.
		mass: Neutral monoisotopic mass in Da.

	Returns:
		float: Theoretical precursor m/z.

	Raises:
		ValueError: If precursor_type is not supported.
	"""
	if precursor_type == '[M+H]+':
		return mass + ATOMS_WEIGHT['H']
	elif precursor_type == '[M+2H]2+':
		return (mass + 2 * ATOMS_WEIGHT['H']) / 2
	elif precursor_type == '[M+Na]+': 
		return mass + ATOMS_WEIGHT['Na']
	elif precursor_type == '[M-H]-':
		return mass - ATOMS_WEIGHT['H']
	elif precursor_type == '[M+H-H2O]+' or precursor_type == '[M-H2O+H]+': 
		return mass + ATOMS_WEIGHT['H'] - (ATOMS_WEIGHT['H'] * 2 + ATOMS_WEIGHT['O'])
	elif precursor_type == '[2M+H]+': 
		return 2 * mass + ATOMS_WEIGHT['H']
	elif precursor_type == '[2M-H]-':
		return 2 * mass - ATOMS_WEIGHT['H']
	elif precursor_type == '[M+H-2H2O]+':
		return mass + ATOMS_WEIGHT['H'] - 2 * (ATOMS_WEIGHT['H'] * 2 + ATOMS_WEIGHT['O'])
	elif precursor_type == '[M+H-NH3]+': 
		return mass + ATOMS_WEIGHT['H'] - (ATOMS_WEIGHT['N'] + ATOMS_WEIGHT['H'] * 3)
	elif precursor_type == '[M+H+NH3]+' or precursor_type == '[M+NH4]+':
		return mass + ATOMS_WEIGHT['H'] + (ATOMS_WEIGHT['N'] + ATOMS_WEIGHT['H'] * 3)
	elif precursor_type == '[M+H-CH2O2]+':
		return mass + ATOMS_WEIGHT['H'] - (ATOMS_WEIGHT['C'] + ATOMS_WEIGHT['H'] * 2 + ATOMS_WEIGHT['O'] * 2)
	elif precursor_type == '[M+H-CH4O2]+':
		return mass + ATOMS_WEIGHT['H'] - (ATOMS_WEIGHT['C'] + ATOMS_WEIGHT['H'] * 4 + ATOMS_WEIGHT['O'] * 2)
	elif precursor_type == '[M-H-CO2]-' or precursor_type == '[M-CHO2]-':
		return mass - ATOMS_WEIGHT['H'] - (ATOMS_WEIGHT['C'] + ATOMS_WEIGHT['O'] * 2)
	elif precursor_type == '[M-H-H2O]-':
		return mass - ATOMS_WEIGHT['H'] - (ATOMS_WEIGHT['H'] * 2 + ATOMS_WEIGHT['O'])
	else: 
		raise ValueError('Unsupported precursor type: {}'.format(precursor_type))

def mass_calculator(precursor_type, precursor_mz):
	"""Back-calculate neutral monoisotopic mass from observed precursor m/z.

	The inverse of precursor_mz_calculator.

	Args:
		precursor_type: Adduct string, e.g. '[M+H]+'.
		precursor_mz: Observed precursor m/z.

	Returns:
		float: Neutral monoisotopic mass in Da.

	Raises:
		ValueError: If precursor_type is not supported.
	"""
	if precursor_type == '[M+H]+':
		return precursor_mz - ATOMS_WEIGHT['H']
	elif precursor_type == '[M+2H]2+':
		return (precursor_mz * 2 - 2 * ATOMS_WEIGHT['H'])
	elif precursor_type == '[M+Na]+':
		return precursor_mz - ATOMS_WEIGHT['Na']
	elif precursor_type == '[M-H]-':
		return precursor_mz + ATOMS_WEIGHT['H']
	elif precursor_type == '[M+H-H2O]+' or precursor_type == '[M-H2O+H]+':
		return precursor_mz - ATOMS_WEIGHT['H'] + (ATOMS_WEIGHT['H'] * 2 + ATOMS_WEIGHT['O'])
	elif precursor_type == '[2M+H]+':
		return (precursor_mz - ATOMS_WEIGHT['H']) / 2
	elif precursor_type == '[2M-H]-':
		return (precursor_mz + ATOMS_WEIGHT['H']) / 2
	elif precursor_type == '[M+H-2H2O]+':
		return precursor_mz - ATOMS_WEIGHT['H'] + 2 * (ATOMS_WEIGHT['H'] * 2 + ATOMS_WEIGHT['O'])
	elif precursor_type == '[M+H-NH3]+':
		return precursor_mz - ATOMS_WEIGHT['H'] + (ATOMS_WEIGHT['N'] + ATOMS_WEIGHT['H'] * 3)
	elif precursor_type == '[M+H+NH3]+' or precursor_type == '[M+NH4]+':
		return precursor_mz - ATOMS_WEIGHT['H'] - (ATOMS_WEIGHT['N'] + ATOMS_WEIGHT['H'] * 3)
	elif precursor_type == '[M+H-CH2O2]+':
		return precursor_mz - ATOMS_WEIGHT['H'] + (ATOMS_WEIGHT['C'] + ATOMS_WEIGHT['H'] * 2 + ATOMS_WEIGHT['O'] * 2)
	elif precursor_type == '[M+H-CH4O2]+':
		return precursor_mz - ATOMS_WEIGHT['H'] + (ATOMS_WEIGHT['C'] + ATOMS_WEIGHT['H'] * 4 + ATOMS_WEIGHT['O'] * 2)
	elif precursor_type == '[M-H-CO2]-' or precursor_type == '[M-CHO2]-':
		return precursor_mz + ATOMS_WEIGHT['H'] + (ATOMS_WEIGHT['C'] + ATOMS_WEIGHT['O'] * 2)
	elif precursor_type == '[M-H-H2O]-':
		return precursor_mz + ATOMS_WEIGHT['H'] + (ATOMS_WEIGHT['H'] * 2 + ATOMS_WEIGHT['O'])
	else:
		raise ValueError('Unsupported precursor type: {}'.format(precursor_type))

if __name__ == '__main__':
	# test sdf2mgf
	spectra = sdf2mgf('../data/origin/Agilent_Combined.sdf', 'agilent_combine')
	print('Number of spectra: {}'.format(len(spectra)))
	for spec in spectra: 
		if spec['params']['precursor_type'] == '[M-H]-':
			print(spec)
			break

	# test filter_spec
	config = {
		'instrument_type': 'ESI-QTOF',
		'instrument': ['Unknown'],
		'ms_level': '2',
		'atom_type': ['C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I', 'Na', 'K'],
		'precursor_type': ['[M+H]+', '[M-H]-'],
		'min_mz': 50,
		'max_mz': 1500,
		'min_peak_num': 5,
		'max_atom_num': 300,
		'min_atom_num': 10,
		'ppm_tolerance': 10
	}
	type2charge = {
		'[M+H]+': 1,
		'[M-H]-': -1
	}
	clean_spectra, smiles_list = filter_spec(spectra, config, type2charge)
	print('Number of clean spectra: {}'.format(len(clean_spectra)))
	print('Number of smiles: {}'.format(len(smiles_list)))
	for spec in clean_spectra: 
		if spec['params']['precursor_type'] == '[M-H]-':
			print(spec)
			break
