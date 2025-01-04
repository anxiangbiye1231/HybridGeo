# %%
from torch import nn
import torch
import torch_geometric
from torch_geometric.nn import global_add_pool
from TDAW import TDAW, TDAW_E

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out) # Oe
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr

class Prot3DGraphModel(nn.Module):
    def __init__(self, d_vocab=21, d_embed=20, d_dihedrals=6, d_pretrained_emb=1280, d_edge=39, d_gcn=[128, 256, 256]):
        super(Prot3DGraphModel, self).__init__()
        d_gcn_in = d_gcn[0]
        self.embed = nn.Embedding(d_vocab, d_embed)
        self.proj_node = nn.Linear(d_embed + d_dihedrals + d_pretrained_emb, d_gcn_in)
        self.proj_edge = nn.Linear(d_edge, d_gcn_in)
        gcn_layer_sizes = [d_gcn_in] + d_gcn
        layers = []
        for i in range(len(gcn_layer_sizes) - 1):
            layers.append((
                torch_geometric.nn.TransformerConv(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in),
                'x, edge_index, edge_attr -> x'
            ))
            layers.append(nn.LeakyReLU())

        self.gcn = torch_geometric.nn.Sequential(
            'x, edge_index, edge_attr', layers)
        self.pool = torch_geometric.nn.global_mean_pool

    def forward(self, data):
        x, edge_index = data.seq, data.edge_index
        batch = data.batch

        x = self.embed(x)
        s = data.node_s
        emb = data.seq_emb
        x = torch.cat([x, s, emb], dim=-1)

        edge_attr = data.edge_s

        x = self.proj_node(x)
        edge_attr = self.proj_edge(edge_attr)

        x = self.gcn(x, edge_index, edge_attr)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        return x

class HG(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=3, residual=True, attention=False, normalize=True, tanh=False):
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(HG, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.lin_node = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf), nn.SiLU())

        self.gconv1 = TDAW(self.hidden_nf, self.hidden_nf)
        self.gconv2 = TDAW(self.hidden_nf, self.hidden_nf)
        self.gconv3 = TDAW(self.hidden_nf, self.hidden_nf)
        #########################################
        self.gnconv1 = TDAW_E(self.hidden_nf, self.hidden_nf)
        self.gnconv2 = TDAW_E(self.hidden_nf, self.hidden_nf)
        self.gnconv3 = TDAW_E(self.hidden_nf, self.hidden_nf)
        self.fc_atom = FC(self.hidden_nf*2, self.hidden_nf, 3, 0.1, self.hidden_nf)
        self.fc_add = FC(self.hidden_nf, self.hidden_nf, 3, 0.1, self.hidden_nf)
        self.fc_p = FC(self.hidden_nf*2, self.hidden_nf, 3, 0.1, self.hidden_nf)
        #########################################
        self.fc = FC(self.hidden_nf*2, self.hidden_nf, 3, 0.1, out_node_nf)

        self.mlp_node_cov = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_nf))
        self.mlp_node_ncov = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_nf))

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))

        self.prot_model = Prot3DGraphModel(d_pretrained_emb=1280, d_gcn=[128, 256, 256])
    def forward(self, drug, pock, comp, esm_fea, edge_attr=None):
        # drug
        h_l, edge_intra_l, edge_attr_l, x_l, batch_l = drug.x, drug.edge_index, drug.edge_attr, drug.pos, drug.batch
        # pock
        h_p, edge_intra_p, edge_attr_p, x_p, batch_p = pock.x, pock.edge_index, pock.edge_attr, pock.pos, pock.batch
        # comp
        h_c, edge_intra_c, edge_inter_c, x_c, batch_c = comp.x, comp.edge_index_intra, comp.edge_index_inter, comp.pos, comp.batch
        edges_c = torch.cat([edge_intra_c, edge_inter_c], dim=-1)
        pos = x_c
        # heterogeneous interaction
        h_c = self.embedding_in(h_c)
        h_intra, h_inter = h_c, h_c
        x_intra, x_inter = x_c, x_c
        for i in range(0, self.n_layers):
            h_intra, x_intra, _ = self._modules["gcl_%d" % i](h_intra, edge_intra_c, x_intra, edge_attr=edge_attr)
            h_inter, x_inter, _ = self._modules["gcl_%d" % i](h_inter, edge_inter_c, x_inter, edge_attr=edge_attr)
        h_c = self.lin_node(self.mlp_node_cov(h_intra) + self.mlp_node_ncov(h_inter))
        h_c = self.gconv1(h_c, edge_intra_c, edge_inter_c, pos)
        h_c = self.gconv2(h_c, edge_intra_c, edge_inter_c, pos)
        h_c = self.gconv3(h_c, edge_intra_c, edge_inter_c, pos)
        h_c = global_add_pool(h_c, batch_c)
        # pock graph transformer
        h_p_new = self.prot_model(esm_fea)

        h = self.fc(torch.cat([h_c, h_p_new], dim=-1))

        return h.view(-1)

class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges

def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr

if __name__ == '__main__':
    pass
    # Dummy parameters
    # batch_size = 8
    # n_nodes = 4
    # n_feat = 1
    # x_dim = 3
    #
    # # Dummy variables h, x and fully connected edges
    # h = torch.ones(batch_size *  n_nodes, n_feat)
    # x = torch.ones(batch_size * n_nodes, x_dim)
    # edges, edge_attr = get_edges_batch(n_nodes, batch_size)
    #
    # # Initialize EGNN
    # egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)
    #
    # # Run EGNN
    # h, x = egnn(h, x, edges, edge_attr)
# %%