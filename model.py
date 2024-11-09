
import torch.nn as nn
import torch.nn.functional as F
from smiles_feature import *

from utils.dataset import MoleculeDataset
from chemprop.features import mol2graph
import math
import torch
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch_geometric.nn import global_mean_pool, GlobalAttention
from models_lib.gnn_model import MPNEncoder
from models_lib.gem_model import GeoGNNModel
from models_lib.seq_model import TrfmSeq2seq

loss_type = {'class': nn.BCEWithLogitsLoss(reduction="none"), 'reg': nn.MSELoss(reduction="none")}


class Global_Attention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.at = GlobalAttention(gate_nn=torch.nn.Linear(hidden_size, 1))

    def forward(self, x, batch):

        return self.at(x, batch)

class WeightFusion(nn.Module):

    def __init__(self, feat_views, feat_dim, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(WeightFusion, self).__init__()
        self.feat_views = feat_views
        self.feat_dim = feat_dim
        self.weight = Parameter(torch.empty((1, 1, feat_views), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(int(feat_dim), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:

        return sum([input[i]*weight for i, weight in enumerate(self.weight[0][0])]) + self.bias


class Fingerprint(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim,
                 fingerprint_dim, output_units_num, p_dropout, feature_dicts,
                 seq_data, data_3d, datas, seq_mask, args, device, compound_encoder_config):
        super(Fingerprint, self).__init__()
        # graph attention for atom embedding
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)  # params 0 1
        self.neighbor_fc = nn.Linear(input_feature_dim + input_bond_dim, fingerprint_dim)  # params 2 3
        self.GRUCell = nn.ModuleList(
            [nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)])  # params 4 5 6 7    8 9 10 11
        self.align = nn.ModuleList([nn.Linear(2 * fingerprint_dim, 1) for r in range(radius)])  # params 12 13   14 15
        self.attend = nn.ModuleList(
            [nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)])  # params 16 17   18 19
        # graph attention for molecule embedding
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)  # params 20 21 22 23
        self.mol_align = nn.Linear(2 * fingerprint_dim, 1)  # params 24 25
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)  # params 26  27
        # you may alternatively assign a different set of parameter in each attentive layer for molecule embedding like in atom embedding process.
        #         self.mol_GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for t in range(T)])
        #         self.mol_align = nn.ModuleList([nn.Linear(2*fingerprint_dim,1) for t in range(T)])
        #         self.mol_attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for t in range(T)])

        self.dropout = nn.Dropout(p=p_dropout)
        ##删除了全链接层的参数，也就是现在params实际上只有36个，而非原版的38个
#        self.output = nn.Linear(fingerprint_dim * 2, output_units_num)  # params 28 29

        self.mol_align2 = nn.Linear(2 * fingerprint_dim, 1)  # params 30 31
        self.mol_attend2 = nn.Linear(fingerprint_dim, fingerprint_dim)  # params 32  33
        self.mol_GRUCell2 = nn.GRUCell(fingerprint_dim, fingerprint_dim)  # params 34 35 36 37
        self.radius = radius
        self.T = T
        self.feature_dicts = feature_dicts
        self.seq_data = seq_data
        self.geo_data = data_3d
        self.gnn_data = datas
        self.seq_mask = seq_mask
        self.args = args
        self.device = device

        self.latent_dim = args.latent_dim
        self.batch_size = args.batch_size
        self.graph = args.graph
        self.sequence = args.sequence
        self.geometry = args.geometry
        
        #Multimodal部分
        # CMPNN
        self.gnn = MPNEncoder(atom_fdim=args.gnn_atom_dim, bond_fdim=args.gnn_bond_dim,
                              hidden_size=args.gnn_hidden_dim, bias=args.bias, depth=args.gnn_num_layers,
                              dropout=args.dropout, activation=args.gnn_activation, device=device)
        # Transformer
        self.transformer = TrfmSeq2seq(input_dim=args.seq_input_dim, hidden_size=args.seq_hidden_dim,
                                       num_head=args.seq_num_heads, n_layers=args.seq_num_layers, dropout=args.dropout,
                                       vocab_num=args.vocab_num, device=device, recons=args.recons).to(self.device)
        # Geometric GNN
        self.compound_encoder = GeoGNNModel(args, compound_encoder_config, device)

        if args.pro_num == 3:
            self.pro_seq = nn.Sequential(nn.Linear(args.seq_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_gnn = nn.Sequential(nn.Linear(args.gnn_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_geo = nn.Sequential(nn.Linear(args.geo_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
        elif args.pro_num == 1:
            self.pro_seq = nn.Sequential(nn.Linear(args.seq_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_gnn = self.pro_seq
            self.pro_geo = self.pro_seq
        #根据任务类型定义损失
        self.entropy = loss_type[args.task_type]

        if args.pool_type == 'mean':
            self.pool = global_mean_pool
        else:
            self.pool = Global_Attention(args.seq_hidden_dim).to(self.device)

        # Fusion
        fusion_dim = args.gnn_hidden_dim * self.graph + args.seq_hidden_dim * self.sequence + \
                     args.geo_hidden_dim * self.geometry
        if self.args.fusion == 3:
            fusion_dim /= (self.graph + self.sequence + self.geometry)  #还是256
#            self.fusion = WeightFusion(self.graph + self.sequence + self.geometry, fusion_dim, device=self.device)
        elif self.args.fusion == 2 or self.args.fusion == 0:
            fusion_dim = args.seq_hidden_dim

        self.dropout = nn.Dropout(args.dropout)
        #输出层
        # MLP Classifier
        self.output_layer = nn.Sequential(nn.Linear(int(fusion_dim), int(fusion_dim)), nn.ReLU(),
                                          nn.Dropout(args.dropout),
                                          nn.Linear(int(fusion_dim), args.output_dim)).to(self.device)

        self.fusion_fi = WeightFusion(self.args.graph + self.args.sequence + self.args.geometry+1,  ###2024/6/4 多了个1
                                   self.args.gnn_hidden_dim, device=self.device).to(self.device)

    def metaGAT(self, smiles_list, params):
        params = [p.to(self.device) for p in params]

        x_atom, x_bonds, \
            x_atom_index, x_bond_index, \
            x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list, self.feature_dicts)
        # print(smiles_list)
        # print(x_atom, x_bonds, \
        # x_atom_index, x_bond_index, \
        # x_mask, smiles_to_rdkit_list)
        atom_list, bond_list, \
            atom_degree_list, bond_degree_list, atom_mask = \
            torch.Tensor(x_atom).to(self.device), torch.Tensor(x_bonds).to(self.device), \
                torch.cuda.LongTensor(x_atom_index), torch.cuda.LongTensor(x_bond_index), \
                torch.Tensor(x_mask).to(self.device)

        atom_mask = atom_mask.unsqueeze(2).to(self.device)
        batch_size, mol_length, num_atom_feat = atom_list.size()
        atom_feature = F.leaky_relu(F.linear(atom_list, params[0], params[1])).to(self.device)

        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0).to(self.device)
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0).to(self.device)
        # then concatenate them
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor], dim=-1)
        neighbor_feature = F.leaky_relu(F.linear(neighbor_feature, params[2], params[3])).to(self.device)

        # generate mask to eliminate the influence of blank atoms
        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length - 1] = 1
        attend_mask[attend_mask == mol_length - 1] = 0
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1).to(self.device)

        softmax_mask = atom_degree_list.clone()
        softmax_mask[softmax_mask != mol_length - 1] = 0
        softmax_mask[softmax_mask == mol_length - 1] = -9e8  # make the softmax value extremly small
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1).to(self.device)

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num,
                                                                fingerprint_dim)
        feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)

        # self.align[0]
        # align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))
        align_score = F.leaky_relu(F.linear(self.dropout(feature_align), params[12], params[13]))
        #             print(attention_weight)
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score, -2)
        #             print(attention_weight)
        attention_weight = attention_weight * attend_mask
        #         print(attention_weight)
        # self.attend[0]
        # neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
        neighbor_feature_transform = F.linear(self.dropout(neighbor_feature), params[16], params[17])
        #             print(features_neighbor_transform.shape)
        context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
        #             print(context.shape)
        context = F.elu(context)
        context_reshape = context.view(batch_size * mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(batch_size * mol_length, fingerprint_dim)

        # self.GRUCell[0]   params 4 5 6 7
        r = torch.sigmoid(F.linear(context_reshape, params[4][:fingerprint_dim], params[6][:fingerprint_dim]) +
                          F.linear(atom_feature_reshape, params[5][:fingerprint_dim], params[7][:fingerprint_dim]))
        z = torch.sigmoid(F.linear(context_reshape, params[4][fingerprint_dim:fingerprint_dim * 2],
                                   params[6][fingerprint_dim:fingerprint_dim * 2]) +
                          F.linear(atom_feature_reshape, params[5][fingerprint_dim:fingerprint_dim * 2],
                                   params[7][fingerprint_dim:fingerprint_dim * 2]))
        n = torch.tanh(F.linear(context_reshape, params[4][fingerprint_dim * 2:], params[6][fingerprint_dim * 2:]) +
                       torch.mul(r, (F.linear(atom_feature_reshape, params[5][fingerprint_dim * 2:],
                                              params[7][fingerprint_dim * 2:]))))
        atom_feature_reshape = torch.mul((1 - z), n) + torch.mul(atom_feature_reshape, z)

        atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

        # do nonlinearity
        activated_features = F.relu(atom_feature)

        for d in range(self.radius - 1):
            # bonds_indexed = [bond_list[i][torch.cuda.LongTensor(bond_degree_list)[i]] for i in range(batch_size)]
            neighbor_feature = [activated_features[i][atom_degree_list[i]] for i in range(batch_size)]

            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num,
                                                                          fingerprint_dim)

            feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)

            # self.align[1]
            align_score = F.leaky_relu(F.linear(self.dropout(feature_align), params[14], params[15]))
            #             print(attention_weight)
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score, -2)
            #             print(attention_weight)
            attention_weight = attention_weight * attend_mask
            #             print(attention_weight)

            # self.attend[1]
            neighbor_feature_transform = F.linear(self.dropout(neighbor_feature), params[18], params[19])
            #             print(features_neighbor_transform.shape)
            context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
            #             print(context.shape)
            context = F.elu(context)
            context_reshape = context.view(batch_size * mol_length, fingerprint_dim)
            #             atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)

            #   self.GRUCell[1]  8 9 10 11
            # atom_feature_reshape222 = self.GRUCell[d + 1](context_reshape, atom_feature_reshape)
            r = torch.sigmoid(F.linear(context_reshape, params[8][:fingerprint_dim], params[10][:fingerprint_dim]) +
                              F.linear(atom_feature_reshape, params[9][:fingerprint_dim], params[11][:fingerprint_dim]))
            z = torch.sigmoid(F.linear(context_reshape, params[8][fingerprint_dim:fingerprint_dim * 2],
                                       params[10][fingerprint_dim:fingerprint_dim * 2]) +
                              F.linear(atom_feature_reshape, params[9][fingerprint_dim:fingerprint_dim * 2],
                                       params[11][fingerprint_dim:fingerprint_dim * 2]))
            n = torch.tanh(
                F.linear(context_reshape, params[8][fingerprint_dim * 2:], params[10][fingerprint_dim * 2:]) +
                torch.mul(r, (
                    F.linear(atom_feature_reshape, params[9][fingerprint_dim * 2:], params[11][fingerprint_dim * 2:]))))
            atom_feature_reshape = torch.mul((1 - z), n) + torch.mul(atom_feature_reshape, z)

            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

            # do nonlinearity
            activated_features = F.relu(atom_feature)

        mol_feature = torch.sum(activated_features * atom_mask, dim=-2)

        # do nonlinearity
        activated_features_mol = F.relu(mol_feature)

        activated_features_mol2 = activated_features_mol.clone()
        mol_feature2 = mol_feature.clone()

        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.type(torch.cuda.FloatTensor)

        for t in range(self.T):
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, mol_length, fingerprint_dim)
            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)

            # self.mol_align
            mol_align_score = F.leaky_relu(F.linear(mol_align, params[24], params[25]))
            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score, -2)
            mol_attention_weight = mol_attention_weight * atom_mask
            #             print(mol_attention_weight.shape,mol_attention_weight)

            # self.mol_attend
            activated_features_transform = F.linear(self.dropout(activated_features), params[26], params[27])
            #             aggregate embeddings of atoms in a molecule
            mol_context = torch.sum(torch.mul(mol_attention_weight, activated_features_transform), -2)
            #             print(mol_context.shape,mol_context)
            mol_context = F.elu(mol_context)

            #   self.mol_GRUCell 20 21 22 23
            # mol_feature222 = self.mol_GRUCell(mol_context, mol_feature)
            r = torch.sigmoid(F.linear(mol_context, params[20][:fingerprint_dim], params[22][:fingerprint_dim]) +
                              F.linear(mol_feature, params[21][:fingerprint_dim], params[23][:fingerprint_dim]))
            z = torch.sigmoid(F.linear(mol_context, params[20][fingerprint_dim:fingerprint_dim * 2],
                                       params[22][fingerprint_dim:fingerprint_dim * 2]) +
                              F.linear(mol_feature, params[21][fingerprint_dim:fingerprint_dim * 2],
                                       params[23][fingerprint_dim:fingerprint_dim * 2]))
            n = torch.tanh(F.linear(mol_context, params[20][fingerprint_dim * 2:], params[22][fingerprint_dim * 2:]) +
                           torch.mul(r, (F.linear(mol_feature, params[21][fingerprint_dim * 2:],
                                                  params[23][fingerprint_dim * 2:]))))
            mol_feature = torch.mul((1 - z), n) + torch.mul(mol_feature, z)

            #             print(mol_feature.shape,mol_feature)

            # do nonlinearity
            activated_features_mol = F.relu(mol_feature)

        activated_features_mol_reverse = torch.flip(activated_features_mol2, dims=[0])
        activated_features_reverse = torch.flip(activated_features, dims=[0])
        mol_softmax_mask_reverse = torch.flip(mol_softmax_mask.clone(), dims=[0])
        atom_mask_reverse = torch.flip(atom_mask.clone(), dims=[0])
        mol_feature_reverse = torch.flip(mol_feature2.clone(), dims=[0])
        for t in range(self.T):
            mol_prediction_expand = activated_features_mol_reverse.unsqueeze(-2).expand(batch_size, mol_length,
                                                                                        fingerprint_dim)
            mol_align = torch.cat([mol_prediction_expand, activated_features_reverse], dim=-1)

            # self.mol_align2
            mol_align_score = F.leaky_relu(F.linear(mol_align, params[28], params[29]))
            mol_align_score = mol_align_score + mol_softmax_mask_reverse
            mol_attention_weight = F.softmax(mol_align_score, -2)
            mol_attention_weight = mol_attention_weight * atom_mask_reverse
            #             print(mol_attention_weight.shape,mol_attention_weight)

            # self.mol_attend2
            activated_features_transform = F.linear(self.dropout(activated_features_reverse), params[30], params[31])
            #             aggregate embeddings of atoms in a molecule
            mol_context = torch.sum(torch.mul(mol_attention_weight, activated_features_transform), -2)
            #             print(mol_context.shape,mol_context)
            mol_context = F.elu(mol_context)

            #   self.mol_GRUCell2
            # mol_feature222 = self.mol_GRUCell(mol_context, mol_feature)
            r = torch.sigmoid(F.linear(mol_context, params[32][:fingerprint_dim], params[34][:fingerprint_dim]) +
                              F.linear(mol_feature_reverse, params[33][:fingerprint_dim], params[35][:fingerprint_dim]))
            z = torch.sigmoid(F.linear(mol_context, params[32][fingerprint_dim:fingerprint_dim * 2],
                                       params[34][fingerprint_dim:fingerprint_dim * 2]) +
                              F.linear(mol_feature_reverse, params[33][fingerprint_dim:fingerprint_dim * 2],
                                       params[35][fingerprint_dim:fingerprint_dim * 2]))
            n = torch.tanh(F.linear(mol_context, params[32][fingerprint_dim * 2:], params[34][fingerprint_dim * 2:]) +
                           torch.mul(r, (F.linear(mol_feature_reverse, params[33][fingerprint_dim * 2:],
                                                  params[35][fingerprint_dim * 2:]))))
            mol_feature_reverse = torch.mul((1 - z), n) + torch.mul(mol_feature_reverse, z)

            # do nonlinearity
            activated_features_mol_reverse = F.relu(mol_feature_reverse)

        mol_feature_all = torch.cat([mol_feature, mol_feature_reverse], dim=1)
        #来自Meta-GAT的融合预测结果
        return mol_feature_all
#SGGRL中的main包括了此项，源代码是在main函数中进行传参的，
    def prepare_data(self, idx, seq_data, seq_mask, gnn_data, geo_data, device):
        idx = [i.item() for i in idx]
        edge_batch1, edge_batch2 = [], []
        geo_gen = geo_data.get_batch(idx)
        node_id_all = [geo_gen[0].batch, geo_gen[1].batch]
        for i in range(geo_gen[0].num_graphs):
            edge_batch1.append(torch.ones(geo_gen[0][i].edge_index.shape[1], dtype=torch.long).to(device) * i)
            edge_batch2.append(torch.ones(geo_gen[1][i].edge_index.shape[1], dtype=torch.long).to(device) * i)
        edge_id_all = [torch.cat(edge_batch1), torch.cat(edge_batch2)]
        # 2D data
        mol_batch = MoleculeDataset([gnn_data[i] for i in idx])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        gnn_batch = mol2graph(smiles_batch, self.args)
        batch_mask_seq, batch_mask_gnn = list(), list()
        for i, (smile, mol) in enumerate(zip(smiles_batch, mol_batch.mols())):
            batch_mask_seq.append(torch.ones(len(smile), dtype=torch.long).to(device) * i)
            batch_mask_gnn.append(torch.ones(mol.GetNumAtoms(), dtype=torch.long).to(device) * i)
        batch_mask_seq = torch.cat(batch_mask_seq)
        batch_mask_gnn = torch.cat(batch_mask_gnn)
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch]).to(device)
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]).to(device)
        return seq_data[idx], seq_mask[idx], batch_mask_seq, gnn_batch, features_batch, batch_mask_gnn, geo_gen, \
            node_id_all, edge_id_all, mask, targets
#SGGRLmulti_modal中的forward函数
    def SGGRL_model(self, idx):
        (seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch,
         features_batch, gnn_batch_batch, geo_gen, node_id_all,
         edge_id_all, mask, targets) = self.prepare_data(idx, self.seq_data, self.seq_mask,
                                                         self.gnn_data, self.geo_data, self.device)
        (trans_batch_seq, seq_mask, batch_mask_seq, gnn_batch_graph, gnn_feature_batch, batch_mask_gnn,
         graph_dict, node_id_all, edge_id_all) = \
            (seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch, features_batch, gnn_batch_batch,
                                                  geo_gen, node_id_all, edge_id_all)
        x_list = list()
        cl_list = list()
        graph_gnn_x = None
        graph_seq_x = None
        graph_geo_x = None
        if self.graph:
            node_gnn_x = self.gnn(gnn_batch_graph, gnn_feature_batch, batch_mask_gnn)
            graph_gnn_x = self.pool(node_gnn_x, batch_mask_gnn)
            if self.args.norm:
                x_list.append(F.normalize(graph_gnn_x, p=2, dim=1))
            else:
                x_list.append(graph_gnn_x)
            cl_list.append(self.pro_gnn(graph_gnn_x))

        if self.sequence:
            nloss, node_seq_x = self.transformer(trans_batch_seq)
            graph_seq_x = self.pool(node_seq_x[seq_mask], batch_mask_seq)
            if self.args.norm:
                x_list.append(F.normalize(graph_seq_x, p=2, dim=1))
            else:
                x_list.append(graph_seq_x)
            cl_list.append(self.pro_seq(graph_seq_x))
        
        if self.geometry:
            node_repr = self.compound_encoder(graph_dict[0], graph_dict[1], node_id_all, edge_id_all)
            graph_geo_x = self.pool(node_repr, node_id_all[0])
            if self.args.norm:
                x_list.append(F.normalize(graph_geo_x, p=2, dim=1))
            else:
                x_list.append(graph_geo_x)
            cl_list.append(self.pro_geo(graph_geo_x.to(self.device)))
            
            #6/8
            
#        if self.args.fusion == 1:
#            molecule_emb = torch.cat([temp for temp in x_list], dim=1)
#        elif self.args.fusion == 2:
#            molecule_emb = x_list[0].mul(x_list[1]).mul(x_list[2])
#        elif self.args.fusion == 3:
#            molecule_emb = self.fusion(torch.stack(x_list, dim=0))
#        else:
#            molecule_emb = torch.mean(torch.cat(x_list), dim=0, keepdim=True)
#
#        if not self.args.norm:
#            molecule_emb = self.dropout(molecule_emb)

        # molecule_emb, graph_gnn_x, graph_seq_x, graph_geo_x, x_list \
        #     = self.multi_modal(seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch, features_batch, gnn_batch_batch,
        #                        geo_gen, node_id_all, edge_id_all)
        # all_loss, lab_loss, cl_loss = model.loss_cal(x_list, preds, targets, mask)
        # total_all_loss = all_loss.item() + total_all_loss
        # total_lab_loss = lab_loss.item() + total_lab_loss
        # total_cl_loss = cl_loss.item() + total_cl_loss
#        pred = self.output_layer(molecule_emb)
        #需要加入pred进行整合吗？整合到输出层
        
        #cl_list：经过pro层的投影结果，其余三个是三个模态的预测
        return graph_gnn_x, graph_seq_x, graph_geo_x,cl_list
    
        ####
        ##################################6.5新增项
    def label_loss(self, pred, label):
        loss_mat = F.cross_entropy(pred, label, reduction='none')
        return loss_mat.mean()

    def cl_loss(self, x1, x2, T=0.1):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss1).mean()
        return loss


    def loss_cal(self, x_list, pred, label, alpha=0.08):
        loss1 = self.label_loss(pred, label)
        loss2 = torch.tensor(0, dtype=torch.float).to(self.device)
        modal_num = len(x_list)
        for i in range(modal_num):
            loss2 += self.cl_loss(x_list[i], x_list[i-1])

        return loss1 + alpha * loss2, loss1, loss2
        ####################################
        
#新加入的forward函数，进行meta和SGGRL的整合，综合输出molpred
    def forward(self, smiles_list, idx, params, type=None):
        # for param_idx, param in enumerate(params):
        #     print(f"Parameter {param_idx}: Shape {param.shape}, Requires Grad {param.requires_grad}")
        ###### Meta-GAT learning
        x_list = []
        mol_feature_all = self.metaGAT(smiles_list, params)
        ################################################################################
        # multimodal -- SGGRL
        # molecule_emb: 三个模态融合的向量, 256=gnn_hidden_dim=seq_num_layers=geo_hidden_dim
        # graph_gnn_emb：Graph embedding gnn_hidden_dim
        # graph_seq_emb：sequence embedding seq_num_layers
        # graph_geo_emb: Geometry embedding geo_hidden_dim
        graph_gnn_emb, graph_seq_emb, graph_geo_emb,cl_list= self.SGGRL_model(idx)

        # fusion 根据要融合的向量做append
        # 这里是把四个向量都做了融合，可以选择自己想要的几种模态的向量来做融合
        x_list.append(mol_feature_all.to(self.device))
        x_list.append(graph_gnn_emb)#0.4
        x_list.append(graph_seq_emb)#0.2
        x_list.append(graph_geo_emb)#0.4

        fusion_emb = self.fusion_fi(torch.stack(x_list, dim=0).to(self.device))
        ################################################################################

       # mol_prediction = F.linear(self.dropout(fusion_emb), params[28], params[29])
       
       #使用多模态全连接层进行输出
        mol_prediction = self.output_layer(fusion_emb)
        # if map_save >= 0:
        #     joblib.dump(mol_feature.cpu().detach(), "./paper/tsne_map/"+time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())+"_"+str(map_save)+".pkl")

        return mol_prediction,cl_list
        #最终融合的预测，cl_list用于求对比LOSS，也进行传递
        # return mol_prediction, mol_feature
        # return atom_feature, mol_prediction, mol_feature
