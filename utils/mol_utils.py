import re
import numpy as np

from rdkit.Chem.rdMolDescriptors import CalcMolFormula

# monoisotopic mass
ATOMS_WEIGHT = {
	'C': 12.000000,
	'H': 1.007825,
	'O': 15.994915,
	'N': 14.003074,
	'F': 18.998403,
	'S': 31.972071,
	'Cl': 34.968853,
	'P': 30.973762,
	'B': 10.012937,
	'Br': 78.918337,
	'I': 126.904473,
	'Na': 22.989770,
	'K': 38.963707,
}
ATOMS_VALENCE = {
	'C': 4,
	'H': 1,
	'O': 2,
	'N': 3,
	'F': 1,
	'S': 2,
	'Cl': 1,
	'P': 3,
	'B': 3,
	'Br': 1,
	'I': 1,
	'Na': 1, 
	'K': 1,
}
ATOMS_INDEX = {
	'C': 0, 
	'H': 1, 
	'O': 2,
	'N': 3,
	'F': 4,
	'S': 5,
	'Cl': 6,
	'P': 7,
	'B': 8,
	'Br': 9,
	'I': 10,
	'Na': 11,
	'K': 12,
}
ATOMS_INDEX_re = {
	0: 'C',
	1: 'H',
	2: 'O',
	3: 'N',
	4: 'F',
	5: 'S',
	6: 'Cl',
	7: 'P',
	8: 'B',
	9: 'Br',
	10: 'I',
	11: 'Na',
	12: 'K'
}

def monoisotopic_mass_calculator(x, mode):
	"""Calculate the monoisotopic mass from a molecule or formula string.

	Args:
		x: RDKit Mol object (mode='mol') or molecular formula string (mode='f').
		mode: 'mol' to accept an RDKit Mol, 'f' to accept a formula string.

	Returns:
		float: Monoisotopic mass in Da.
	"""
	assert mode in ['mol', 'f'], 'Invalid mode: {}'.format(mode)
	 
	if mode == 'mol': 
		x = CalcMolFormula(x)
	
	f_dict = formula_to_dict(x)
	iso_mass = np.sum([ATOMS_WEIGHT.get(k, 0) * v for k, v in f_dict.items()]).item()
	return iso_mass

def dict_to_formula(formula_dict):
	"""Convert an atom-count dict back to a molecular formula string.

	Args:
		formula_dict: Mapping of atom symbol to count, e.g. {'C': 10, 'H': 12}.

	Returns:
		str: Molecular formula string, e.g. 'C10H12'.
	"""
	formula = ''
	for k, v in formula_dict.items():
		if v == 1:
			formula += k
		elif v > 1:
			formula += k + str(v)
	return formula

def formula_to_dict(formula):
	"""Parse a molecular formula string into an atom-count dict.

	Args:
		formula: Molecular formula string, e.g. 'C10H12N2O3'.

	Returns:
		dict: Mapping of atom symbol to count, e.g. {'C': 10, 'H': 12, ...}.
		      Returns {} if formula is not a string.
	"""
	if not isinstance(formula, str): # it is possible that all the predicted formula is None in BUDDY & SIRIUS
		return {} 
	atom_counts = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
	formula_dict = {atom: int(count) if count else 1 for atom, count in atom_counts}
	return formula_dict

def formula_to_vector(formula):
	"""Convert a molecular formula to a fixed-length atom-count vector.

	The vector order follows ATOMS_INDEX (C, H, O, N, F, S, Cl, P, B, Br, I, Na, K).

	Args:
		formula: Molecular formula string.

	Returns:
		list[int]: Length-13 vector of atom counts.
	"""
	vector = [0] * len(ATOMS_INDEX)
	formula_dict = formula_to_dict(formula)

	for atom, count in formula_dict.items(): 
		index = ATOMS_INDEX.get(atom, None)
		if index is not None:
			vector[index] = int(count) if count else 1
	return vector

def vector_to_formula(vec, withH=True):
	"""Convert a fixed-length atom-count vector back to a formula string.

	Args:
		vec: Sequence of atom counts aligned with ATOMS_INDEX.
		withH: If False, hydrogen is omitted from the output string.

	Returns:
		str: Molecular formula string.
	"""
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

if __name__ == '__main__': 
	from rdkit import Chem
	# ignore the warning
	from rdkit import RDLogger 
	RDLogger.DisableLog('rdApp.*')

	formula = 'C10H12N2O3'
	vec = formula_to_vector(formula)
	print(formula)
	print(vec)
	print(vector_to_formula(vec))

	smiles = 'CC(C)C1=CC(=C(C=C1)O)C(=O)O'
	mol = Chem.MolFromSmiles(smiles)
	mass = monoisotopic_mass_calculator(mol, mode='mol')
	print(smiles, mass)

	formula = 'C10H12N2O3'
	mass = monoisotopic_mass_calculator(formula, mode='f')
	print(formula, mass)