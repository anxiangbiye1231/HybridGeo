# %%
import os
import warnings
import torch
import pickle
import csv
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore")
# %%

def generate_pocket(data_dir, distance):
    complex_id = os.listdir(data_dir)
    for cid in tqdm(complex_id, ncols=50):
        complex_dir = os.path.join(data_dir, cid)
        lig_native_path = os.path.join(complex_dir, f"{cid}_ligand.mol2")
        protein_path = os.path.join(complex_dir, f"{cid}_protein.pdb")
        if os.path.exists(os.path.join(complex_dir, f'Pocket_{distance}A.pdb')):
            continue

        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(lig_native_path)
        pymol.cmd.remove('hydrogens')
        try:
            pymol.cmd.select('Pocket', f'byres {cid}_ligand around {distance}')
            pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{distance}A.pdb'), 'Pocket')
        except Exception:
            count = count + 1
            print(cid, f'{count} error')
        pymol.cmd.delete('all')

def generate_complex(data_dir, data_df, distance=5, input_ligand_format='mol2'):
    un = []
    pbar = tqdm(total=len(data_df), ncols=50)
    for i, row in data_df.iterrows():
        cid, pKa = row['id'], float(row['affinity'])
        pocket_path = os.path.join(data_dir, cid, f'Pocket_{distance}A.pdb')
        if not os.path.exists(os.path.join(f"{data_dir}/{cid}/{cid}_{distance}A.rdkit")):
            try:
                if input_ligand_format != 'pdb':
                    ligand_input_path = os.path.join(data_dir, cid, f'{cid}_ligand.{input_ligand_format}')
                    ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
                    os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
                else:
                    ligand_path = os.path.join(data_dir, cid, f'{cid}_ligand.pdb')

                save_path = os.path.join(f"{data_dir}/{cid}/{cid}_{distance}A.rdkit")
                ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
                if ligand == None:
                    print(f"Unable to process ligand of {cid}")
                    continue

                pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
                if pocket == None:
                    print(f"Unable to process protein of {cid}")
                    continue
            except:
                un.append(cid)
            complex = (ligand, pocket)
            with open(save_path, 'wb') as f:
                pickle.dump(complex, f)

            pbar.update(1)
    print(un, 'Unprocessed')

def get_Ca_coord(residues):
    residues_coord = []
    for res in residues:
        try:
            N_coord = [round(num, 3) for num in res.child_dict['N'].coord.tolist()]
            CA_coord = [round(num, 3) for num in res.child_dict['CA'].coord.tolist()]
            C_coord = [round(num, 3) for num in res.child_dict['C'].coord.tolist()]
            O_coord = [round(num, 3) for num in res.child_dict['O'].coord.tolist()]
            res_coor = [N_coord, CA_coord, C_coord, O_coord]
        except:
            print(res)
        residues_coord.append(res_coor)
    return residues_coord

def get_sequence(structure):
    residues_list = [residue for residue in structure.get_residues() if residue.get_id()[0] == ' ']

    residues = []
    for res in residues_list:
        if 'N' in res.child_dict.keys() and 'CA' in res.child_dict.keys() and 'C' in res.child_dict.keys() and 'O' in res.child_dict.keys():
            residues.append(res)
    seqs = ''.join([seq1(residue.get_resname()) for residue in residues])
    coords = get_Ca_coord(residues)
    return residues, seqs, coords

def generate_mask(seq_res, sequence, pocket_res):
    mask = len(sequence) * [0]
    pock_f_id = [res_i.full_id for res_i in pocket_res]
    prot_f_id = [res_j.full_id for res_j in seq_res]
    for res_i in pocket_res:
        for res_j in seq_res:
            if res_i.full_id == res_j.full_id:
                index = seq_res.index(res_j)
                mask[index] = 1
    ## valid num
    idx_1_list = []
    for i, ele in enumerate(mask):
        if ele == 1:
            idx_1_list.append(i)
    for id_i in pock_f_id:
        p_f_id = [prot_f_id[id_j] for id_j in idx_1_list]
        if id_i in p_f_id:
            new_mask = mask
        else:
            print(sequence, 'unmatched')
    return mask

def extract_pos(prot_fea, pock_pos):
    indices = torch.nonzero(torch.tensor(pock_pos), as_tuple=True)
    prot_fea_tensor = torch.tensor(prot_fea)
    pock_fea = prot_fea_tensor[indices[0]].numpy()
    return pock_fea

def generate_pock_esm(datasets_path, esm_dir, distance):
    set_list = os.listdir(datasets_path)
    for prot_name in tqdm(set_list, ncols=50):
        prot_file = os.path.join(datasets_path, prot_name + f'/{prot_name}_protein.pdb')
        pock_file = os.path.join(datasets_path, prot_name + f'/Pocket_{distance}A.pdb')

        if os.path.exists(esm_dir + f'prot_fea/{prot_name}.pt'):
            with open(esm_dir + f'prot_fea/{prot_name}.pt', 'rb') as f:
                prot_fea = torch.load(f)['lm_prot_fea']
            if not os.path.exists(esm_dir + f'pock_{distance}A_fea/{prot_name}_pock.pt'):
                try:
                    parser = PDBParser()
                    protein = parser.get_structure(prot_name, prot_file)
                    pocket = parser.get_structure(prot_name, pock_file)

                    prot_res, prot_seq, prot_coords = get_sequence(protein)
                    pock_res, pock_seq, pock_coords = get_sequence(pocket)
                    pock_pos = generate_mask(prot_res, prot_seq, pock_res)
                    pock_fea = extract_pos(prot_fea, pock_pos)
                    torch.save({'lm_pock_fea': pock_fea}, esm_dir + f'pock_{distance}A_fea/{prot_name}_pock.pt')
                except:
                    print(f'{prot_name}', 'error')

if __name__ == '__main__':
    distance = 5
    input_ligand_format = 'mol2'
    data_dir = './data/train'
    esm_dir = './data/esm/'
    data_df = pd.read_csv(os.path.join('./data/valid.csv'))
    ## generate pocket within 5 Ångström around ligand
    generate_pocket(data_dir=data_dir, distance=distance)
    generate_complex(data_dir, data_df, distance=distance, input_ligand_format=input_ligand_format)
    generate_pock_esm(data_dir, esm_dir, distance)
