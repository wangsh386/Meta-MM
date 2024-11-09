import torch
import os
import joblib
import time
import torch.nn as nn
import torch.nn.functional as F
from smiles_feature import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Fingerprint(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim,
                 fingerprint_dim, output_units_num, p_dropout, feature_dicts, feature_dicts_target):
        super(Fingerprint, self).__init__()
        # graph attention for atom embedding
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)   # params 0 1
        self.neighbor_fc = nn.Linear(input_feature_dim + input_bond_dim, fingerprint_dim) # params 2 3
        self.GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)]) # params 4 5 6 7    8 9 10 11
        self.align = nn.ModuleList([nn.Linear(2 * fingerprint_dim, 1) for r in range(radius)])  # params 12 13   14 15
        self.attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)])   # params 16 17   18 19
        # graph attention for molecule embedding
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim) # params 20 21 22 23
        self.mol_align = nn.Linear(2 * fingerprint_dim, 1)  # params 24 25
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)  # params 26  27
        # you may alternatively assign a different set of parameter in each attentive layer for molecule embedding like in atom embedding process.
        #         self.mol_GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for t in range(T)])
        #         self.mol_align = nn.ModuleList([nn.Linear(2*fingerprint_dim,1) for t in range(T)])
        #         self.mol_attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for t in range(T)])

        self.dropout = nn.Dropout(p=p_dropout)
        self.output = nn.Linear(fingerprint_dim, output_units_num) # params 28 29

        self.radius = radius
        self.T = T
        self.feature_dicts = feature_dicts
        self.feature_dicts_target = feature_dicts_target

    def forward(self, smiles_list, params, type="train"):

        params = [p.to(device) for p in params]

        if "test_tranfer".__eq__(type):
            x_atom, x_bonds, \
            x_atom_index, x_bond_index, \
            x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list, self.feature_dicts_target)
        else:
            x_atom, x_bonds, \
            x_atom_index, x_bond_index, \
            x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list, self.feature_dicts)
        # print(smiles_list)
        # print(x_atom, x_bonds, \
        # x_atom_index, x_bond_index, \
        # x_mask, smiles_to_rdkit_list)

        q = torch.Tensor(x_atom).to(device)
        atom_list, bond_list, \
        atom_degree_list, bond_degree_list, atom_mask = \
                                torch.Tensor(x_atom).to(device), torch.Tensor(x_bonds).to(device),\
                                torch.cuda.LongTensor(x_atom_index), torch.cuda.LongTensor(x_bond_index),\
                                torch.Tensor(x_mask).to(device)

        atom_mask = atom_mask.unsqueeze(2)
        batch_size, mol_length, num_atom_feat = atom_list.size()
        atom_feature = F.leaky_relu(F.linear(atom_list, params[0], params[1]))

        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)
        # then concatenate them
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor], dim=-1)
        neighbor_feature = F.leaky_relu(F.linear(neighbor_feature, params[2], params[3]))

        # generate mask to eliminate the influence of blank atoms
        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length - 1] = 1
        attend_mask[attend_mask == mol_length - 1] = 0
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = atom_degree_list.clone()
        softmax_mask[softmax_mask != mol_length - 1] = 0
        softmax_mask[softmax_mask == mol_length - 1] = -9e8  # make the softmax value extremly small
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

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
                          F.linear(atom_feature_reshape, params[5][:fingerprint_dim],params[7][:fingerprint_dim]))
        z = torch.sigmoid(F.linear(context_reshape, params[4][fingerprint_dim:fingerprint_dim*2], params[6][fingerprint_dim:fingerprint_dim*2]) +
                          F.linear(atom_feature_reshape, params[5][fingerprint_dim:fingerprint_dim*2], params[7][fingerprint_dim:fingerprint_dim*2]))
        n = torch.tanh(F.linear(context_reshape, params[4][fingerprint_dim*2:], params[6][fingerprint_dim*2:]) +
                       torch.mul(r, (F.linear(atom_feature_reshape, params[5][fingerprint_dim*2:], params[7][fingerprint_dim*2:]))))
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
            z = torch.sigmoid(F.linear(context_reshape, params[8][fingerprint_dim:fingerprint_dim * 2], params[10][fingerprint_dim:fingerprint_dim * 2]) +
                              F.linear(atom_feature_reshape, params[9][fingerprint_dim:fingerprint_dim * 2], params[11][fingerprint_dim:fingerprint_dim * 2]))
            n = torch.tanh(F.linear(context_reshape, params[8][fingerprint_dim * 2:], params[10][fingerprint_dim * 2:]) +
                           torch.mul(r, (F.linear(atom_feature_reshape, params[9][fingerprint_dim * 2:], params[11][fingerprint_dim * 2:]))))
            atom_feature_reshape = torch.mul((1 - z), n) + torch.mul(atom_feature_reshape, z)

            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

            # do nonlinearity
            activated_features = F.relu(atom_feature)

        mol_feature = torch.sum(activated_features * atom_mask, dim=-2)

        # do nonlinearity
        activated_features_mol = F.relu(mol_feature)

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

            #self.mol_attend
            activated_features_transform = F.linear(self.dropout(activated_features), params[26], params[27])
            #             aggregate embeddings of atoms in a molecule
            mol_context = torch.sum(torch.mul(mol_attention_weight, activated_features_transform), -2)
            #             print(mol_context.shape,mol_context)
            mol_context = F.elu(mol_context)

            #   self.mol_GRUCell 20 21 22 23
            # mol_feature222 = self.mol_GRUCell(mol_context, mol_feature)
            r = torch.sigmoid(F.linear(mol_context, params[20][:fingerprint_dim], params[22][:fingerprint_dim]) +
                              F.linear(mol_feature, params[21][:fingerprint_dim], params[23][:fingerprint_dim]))
            z = torch.sigmoid(F.linear(mol_context, params[20][fingerprint_dim:fingerprint_dim * 2], params[22][fingerprint_dim:fingerprint_dim * 2]) +
                              F.linear(mol_feature, params[21][fingerprint_dim:fingerprint_dim * 2], params[23][fingerprint_dim:fingerprint_dim * 2]))
            n = torch.tanh(F.linear(mol_context, params[20][fingerprint_dim * 2:], params[22][fingerprint_dim * 2:]) +
                           torch.mul(r, (F.linear(mol_feature, params[21][fingerprint_dim * 2:], params[23][fingerprint_dim * 2:]))))
            mol_feature = torch.mul((1 - z), n) + torch.mul(mol_feature, z)

            #             print(mol_feature.shape,mol_feature)

            # do nonlinearity
            activated_features_mol = F.relu(mol_feature)

        mol_prediction = F.linear(self.dropout(mol_feature), params[-2], params[-1])
        # if map_save >= 0:
        #     joblib.dump(mol_feature.cpu().detach(), "./paper/tsne_map/"+time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())+"_"+str(map_save)+".pkl")

        return mol_prediction
        # return atom_feature, mol_prediction, mol_feature