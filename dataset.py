# %%
"""
Adapted from
https://github.com/jingraham/neurips19-graph-protein-design
https://github.com/drorlab/gvp-pytorch
"""
import math
import shutil

import torch.nn.functional as F
import torch_geometric
import os
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import Dataset, DataLoader
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data import Batch, Data
import warnings
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

LETTER_TO_NUM = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12, 'X':20}

NUM_TO_LETTER = {v:k for k, v in LETTER_TO_NUM.items()}

def featurize_pocket_graph(protein, name=None, num_pos_emb=16, num_rbf=16, contact_cutoff=8.,): # 'N', 'CA', 'C', 'O'
    coords = torch.as_tensor(protein['coords'], dtype=torch.float32)
    seq = torch.as_tensor([LETTER_TO_NUM[a] for a in protein['seq']], dtype=torch.long)
    seq_emb = torch.from_numpy(torch.load(protein['embed'])['lm_pock_fea'])

    mask = torch.isfinite(coords.sum(dim=(1,2)))
    coords[~mask] = np.inf

    X_ca = coords[:, 1]
    ca_mask = torch.isfinite(X_ca.sum(dim=(1)))
    ca_mask = ca_mask.float()
    ca_mask_2D = torch.unsqueeze(ca_mask, 0) * torch.unsqueeze(ca_mask, 1)
    dX_ca = torch.unsqueeze(X_ca, 0) - torch.unsqueeze(X_ca, 1)
    D_ca = ca_mask_2D * torch.sqrt(torch.sum(dX_ca**2, 2) + 1e-6)
    edge_index = torch.nonzero((D_ca < contact_cutoff) & (ca_mask_2D == 1))
    edge_index = edge_index.t().contiguous()

    O_feature = _local_frame(X_ca, edge_index)
    pos_embeddings = _positional_embeddings(edge_index, num_embeddings=num_pos_emb)
    E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf)

    dihedrals = _dihedrals(coords)
    orientations = _orientations(X_ca)
    sidechains = _sidechains(coords)

    node_s = dihedrals
    node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
    edge_s = torch.cat([rbf, O_feature, pos_embeddings], dim=-1)
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
            (node_s, node_v, edge_s, edge_v))

    data = torch_geometric.data.Data(x=X_ca, seq=seq, name=name,
                                        node_s=node_s, node_v=node_v,
                                        edge_s=edge_s, edge_v=edge_v,
                                        edge_index=edge_index, mask=mask,
                                        seq_emb=seq_emb)
    return data


def _dihedrals(X, eps=1e-7):
    X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def _positional_embeddings(edge_index,
                            num_embeddings=None,
                            period_range=[2, 1000]):
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def _orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def _local_frame(X, edge_index, eps=1e-6):
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    o_1 = _normalize(u_2 - u_1, dim=-1)
    O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 1)
    O = F.pad(O, (0, 0, 0, 0, 1, 2), 'constant', 0)

    # dX = X[edge_index[0]] - X[edge_index[1]]
    dX = X[edge_index[1]] - X[edge_index[0]]
    dX = _normalize(dX, dim=-1)
    # dU = torch.bmm(O[edge_index[1]], dX.unsqueeze(2)).squeeze(2)
    dU = torch.bmm(O[edge_index[0]], dX.unsqueeze(2)).squeeze(2)
    R = torch.bmm(O[edge_index[0]].transpose(-1,-2), O[edge_index[1]])
    Q = _quaternions(R)
    O_features = torch.cat((dU,Q), dim=-1)

    return O_features


def _quaternions(R):
    # Simple Wikipedia version
    # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:, i, j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)
    return Q

# %%
def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edges(g):
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x)
               for x in (Chem.rdchem.BondType.SINGLE, \
                         Chem.rdchem.BondType.DOUBLE, \
                         Chem.rdchem.BondType.TRIPLE, \
                         Chem.rdchem.BondType.AROMATIC,\
                         Chem.rdchem.BondType.IONIC,\
                         Chem.rdchem.BondType.DATIVE,\
                         Chem.rdchem.BondType.HYDROGEN,\
                         Chem.rdchem.BondType.THREECENTER,\
                         Chem.rdchem.BondType.DATIVEL,\
                         Chem.rdchem.BondType.DATIVER)]
        e_t.append(int(d['IsConjugated'] == False))
        e_t.append(int(d['IsConjugated'] == True))
        e[(n1, n2)] = e_t
    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))
    return edge_index, edge_attr

def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        graph.add_edge(i, j)

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    graph = graph.to_directed()
    # Read Edges
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                graph.add_edge(i, j,
                           b_type=e_ij.GetBondType(),
                           # 1 more edge features 2 dim
                           IsConjugated=int(e_ij.GetIsConjugated()),
                           )
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index, edge_attr = get_edges(graph)

    return x, edge_index, edge_attr

def inter_graph(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+atom_num_l) 

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

    return edge_index_inter

def get_coord(residues):
    residues_coord = []
    for res in residues:
        try:
            N_coord = [round(num, 3) for num in res.child_dict['N'].coord.tolist()]
            CA_coord = [round(num, 3) for num in res.child_dict['CA'].coord.tolist()]
            C_coord = [round(num, 3) for num in res.child_dict['C'].coord.tolist()]
            O_coord = [round(num, 3) for num in res.child_dict['O'].coord.tolist()]
            res_coor = (N_coord, CA_coord, C_coord, O_coord)
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
    coords = get_coord(residues)
    return residues, seqs, coords

def process_pock(prot_name, pock_file, pock_esm):
    parser = PDBParser()
    pocket = parser.get_structure(prot_name, pock_file)
    pock_res, pock_seq, pock_coords = get_sequence(pocket)
    pock_dic = {'name': prot_name, 'seq': pock_seq, 'coords': pock_coords, 'embed': pock_esm}
    return pock_dic

# %%
def mols2graphs(complex_path, label, save_path, cid, pock_path, pock_esm_path, dis_threshold=5.):

    with open(complex_path, 'rb') as f:
        ligand, pocket = pickle.load(f)

    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
    x_l, edge_index_l, edge_attr_l = mol2graph(ligand)
    x_p, edge_index_p, edge_attr_p = mol2graph(pocket)
    # comp fea
    x = torch.cat([x_l, x_p], dim=0)
    edge_index_intra = torch.cat([edge_index_l, edge_index_p+atom_num_l], dim=-1)
    edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
    y = torch.FloatTensor([label])
    pos = torch.concat([pos_l, pos_p], dim=0)
    split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p,))], dim=0)
    # pock fea
    pock_dic = process_pock(cid, pock_path, pock_esm_path)
    pock_new_fea = featurize_pocket_graph(pock_dic, cid, num_pos_emb=16, num_rbf=16, contact_cutoff=8.)

    drug_data = Data(x=x_l, edge_index=edge_index_l, edge_index_inter=edge_index_inter, edge_attr=edge_attr_l, pos=pos_l, y=y)
    pock_data = Data(x=x_p, edge_index=edge_index_p, edge_index_inter=edge_index_inter, edge_attr=edge_attr_p, pos=pos_p, y=y)
    comp_data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, y=y, pos=pos, split=split)
    data = {'drug': drug_data, 'pock': pock_data, 'pock_new': pock_new_fea, 'comp': comp_data}
    torch.save(data, save_path)
    # return data

# %%
class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, dis_threshold=5, graph_type='Graph_HG', num_process=8, create=False):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self._pre_process()

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type
        dis_thresholds = repeat(self.dis_threshold, len(data_df))

        complex_path_list = []
        pKa_list = []
        graph_path_list = []
        cid_list = []
        pock_path_list = []
        pock_esm_path_list = []
        for i, row in tqdm(data_df.iterrows(), ncols=80):
            cid, pKa = row['id'], float(row['affinity'])
            complex_dir = os.path.join(data_dir, cid)

            graph_path = os.path.join(complex_dir, f"{cid}_fea.pyg")
            complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")
            pock_path = os.path.join(complex_dir, f"Pocket_5A.pdb")
            pock_esm_path = os.path.join(complex_dir, f"{cid}_pock.pt")

            # if os.path.exists(complex_path) and os.path.exists(pock_path) and os.path.exists(pock_esm_path):
            if os.path.exists(complex_path) and os.path.exists(pock_path) and os.path.exists(pock_esm_path) and os.path.exists(graph_path):
                complex_path_list.append(complex_path)
                pKa_list.append(pKa)
                graph_path_list.append(graph_path)
                cid_list.append(cid)
                pock_path_list.append(pock_path)
                pock_esm_path_list.append(pock_esm_path)

        if self.create:
            print('Generate complex graph...')
            # multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, pKa_list, graph_path_list, cid_list, pock_path_list, pock_esm_path_list, dis_thresholds))
            pool.close()
            pool.join()

        self.graph_paths = graph_path_list

    def __getitem__(self, idx):
        data = torch.load(self.graph_paths[idx])
        return data

    def collate_fn(self, batch):
        drug_batch = Batch.from_data_list([item['drug'] for item in batch])
        pock_batch = Batch.from_data_list([item['pock'] for item in batch])
        comp_batch = Batch.from_data_list([item['comp'] for item in batch])
        pock_new_batch = Batch.from_data_list([item['pock_new'] for item in batch])
        return drug_batch, pock_batch, comp_batch, pock_new_batch

    def __len__(self):
        return len(self.graph_paths)

if __name__ == '__main__':
    data_root = './data'
    toy_dir = os.path.join(data_root, 'train')
    toy_df = pd.read_csv(os.path.join(data_root, "train.csv"))
    toy_set = GraphDataset(toy_dir, toy_df, graph_type='Graph_HG', dis_threshold=5, create=True)
    train_loader = PLIDataLoader(toy_set, batch_size=128, shuffle=True, num_workers=4)
# %%
