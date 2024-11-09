import json
import os
import sys
from MetaLearner import *
from model import Fingerprint
from smiles_feature import *
from dataset import *
from regdataset import *
from MetaLearnerReg import *
import random
import pandas as pd
import time, datetime
from parser_args import get_args
from utils.dataset import Seq2seqDataset, get_data, InMemoryDataset
from build_vocab import WordVocab
from chemprop.features import get_atom_fdim, get_bond_fdim
import warnings

from featurizers.gem_featurizer import GeoPredTransformFn
device = torch.device('cuda:2')
####2024/6/6 20:50-ROC
# 忽略所有的用户警告
warnings.filterwarnings("ignore", category=UserWarning)
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
os.environ['PYTHONUNBUFFERED'] = '1'
PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4

seed = 188
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

### 准备数据迭代器
k_spt = 2 ## support data 的个数

k_spt_pos = 10   # k-shot positive
k_spt_neg = 10   # k-shot negative
k_query = 16  ## query data 的个数
Nway = 8

#以上几个数字被用于dataset进行支持集等的创建

p_dropout = 0.2

# 此处的维度要和graph_gnn_emb, graph_seq_emb, graph_geo_emb对齐，是它们的一半
fingerprint_dim = 128

dataset_type="class" #class reg
metric="ROC" #ROC accs mae
name = 'sider'  # sider tox21 muv qm9
if dataset_type=="class":
    output_units_num = 2
elif dataset_type=="reg":
    output_units_num = 1
radius = 2
T = 2
episode = 50
# episode = 1
test_batchsz = 20 #20
epochs =100   #50
# epochs = 1

raw_filename = "data/{}.csv".format(name)
if name == 'tox21':
    tasks = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
                  "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
                    "SR-HSE", "SR-MMP", "SR-p53",]
    task_num = 9
    test_task_num = 3
elif name == 'sider':
    raw_filename = "data/sider.csv"
    tasks = ["SIDER1", "SIDER2", "SIDER3", "SIDER4", "SIDER5",
             "SIDER6", "SIDER7", "SIDER8", "SIDER9", "SIDER10",
             "SIDER11", "SIDER12", "SIDER13", "SIDER14", "SIDER15",
             "SIDER16", "SIDER17", "SIDER18", "SIDER19", "SIDER20",
             "SIDER21", "SIDER22", "SIDER23", "SIDER24", "SIDER25", "SIDER26", "SIDER27"]
    task_num = 21
    test_task_num = 6
elif name == 'muv':
    raw_filename = "data/muv.csv"
    tasks = ["MUV-466","MUV-548","MUV-600","MUV-644","MUV-652",
             "MUV-689","MUV-692","MUV-712","MUV-713","MUV-733",
             "MUV-737","MUV-810","MUV-832","MUV-846","MUV-852",
             "MUV-858","MUV-859"]
    task_num = 12
    test_task_num = 5

    #选用lumo,G,cv
elif name == 'qm9':
    raw_filename = "data/qm9.csv"
    tasks = ["A", "B", "C", "mu", "alpha",
             "homo", "lumo", "gap", "r2", "zpve",
             "u0", "u298", "h298", "g298","cv", "u0_atom",
             "u298_atom", "h298_atom", "g298_atom"]
    task_num = 9
    test_task_num = 3

def process_data():
    smiles_tasks_df = pd.read_csv(raw_filename)
    smilesList = smiles_tasks_df.smiles.values
    print("number of all smiles: ",len(smilesList))

    remained_smiles = []
    for smiles in smilesList:
        try:
            remained_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            print(smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))

    feature_filename = raw_filename.replace('.csv','.pickle')
    if os.path.isfile(feature_filename):
        feature_dicts = pickle.load(open(feature_filename, "rb"))
    else:
        feature_dicts = save_smiles_dicts(remained_smiles, feature_filename)
    # feature_dicts = save_smiles_dicts(remained_smiles, feature_filename)
    # print(feature_dicts)
    # print(feature_dicts['smiles_to_atom_mask'])
    #
    smiles_tasks_df['remained_smiles'] = remained_smiles
    remained_df = smiles_tasks_df[smiles_tasks_df["remained_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    print("number of remained  smiles: ", len(remained_df.smiles.values))
    # print(remained_df)
    # remained_df.to_csv('data/sider_remained.csv',index=False)
    remained_df.to_csv('data/{}_remained.csv'.format(name),index=False)
    del remained_df['remained_smiles']
    remained_df.to_csv('data/{}_remained_o.csv'.format(name),index=False)

##################先处理数据#####################
# process_data()
################处理完可以注释#################

remained_filename = 'data/{}_remained.csv'.format(name)
remained_df = pd.read_csv('data/{}_remained.csv'.format(name))
remained_smiles = remained_df['remained_smiles'].tolist()
feature_filename = remained_filename.replace('.csv','.pickle')
if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb"))
else:
    feature_dicts = save_smiles_dicts(remained_smiles, feature_filename)


def load_json_config(path):
    """tbd"""
    return json.load(open(path, 'r'))


def load_smiles_to_dataset(data_path):
    """tbd"""
    data_list = []
    with open(data_path, 'r') as f:
        tmp_data_list = [line.strip() for line in f.readlines()]
        tmp_data_list = tmp_data_list[1:]
    data_list.extend(tmp_data_list)
    dataset = InMemoryDataset(data_list)
    return dataset


#根据数据建立三个模态的数据，也调节了部分args中的参数
def SGGRL(args):
    data_path = 'data/{}_remained_o.csv'.format(name)
    args.dataset = '{}_remained'.format(name)
    datas, args.seq_len = get_data(path=data_path, args=args)
    args.output_dim = args.num_tasks = datas.num_tasks()
    args.gnn_atom_dim = get_atom_fdim(args)
    args.gnn_bond_dim = get_bond_fdim(args) + (not args.atom_messages) * args.gnn_atom_dim
    args.features_size = datas.features_size()

    smiles = datas.smiles()
    vocab = WordVocab.load_vocab('data/{}_remained_o_vocab.pkl'.format(name))
    args.seq_input_dim = args.vocab_num = len(vocab)
    seq = Seq2seqDataset(list(np.array(smiles)), vocab=vocab, seq_len=args.seq_len, device=device)
    seq_data = torch.stack([temp[1] for temp in seq])

    # 3d data process
    compound_encoder_config = load_json_config(args.compound_encoder_config)
    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate

    data_3d = InMemoryDataset(datas.smiles())
    transform_fn = GeoPredTransformFn(model_config['pretrain_tasks'], model_config['mask_ratio'])
    if not os.path.exists('data/{}/'.format(args.dataset)):
        data_3d.transform(transform_fn, num_workers=1)
        data_3d.save_data('data/{}/'.format(args.dataset))
    else:
        data_3d = data_3d._load_npz_data_path('data/{}/'.format(args.dataset))
        data_3d = InMemoryDataset(data_3d)

    data_3d.get_data(device)

    #
    seq_mask = torch.zeros(len(datas), args.seq_len).bool().to(device)
    for i, smile in enumerate(smiles):
        seq_mask[i, 1:1 + len(smile)] = True

    # Multi Modal Init
    args.seq_hidden_dim = args.gnn_hidden_dim
    args.geo_hidden_dim = args.gnn_hidden_dim
    return compound_encoder_config, seq_data, data_3d, datas, seq_mask

# seq data process

args = get_args()
if dataset_type=="reg":
    args.task_type="reg"
compound_encoder_config, seq_data, data_3d, datas, seq_mask = SGGRL(args)


#用于设置训练集和测试集
if dataset_type=="class":
    data_train = MyDataset(remained_df, k_spt, k_spt_pos, k_spt_neg, k_query, tasks[:task_num], Nway, epochs, type="train")
    dataset_train = data.DataLoader(dataset=data_train, batch_size=1, shuffle=True)

    data_test = MyDataset(remained_df, k_spt, k_spt_pos, k_spt_neg, k_query, tasks[task_num:], test_task_num, test_batchsz, type="test")
    dataset_test = data.DataLoader(dataset=data_test, batch_size=1, shuffle=True)

#回归数据集：不需要正负样本，直接进行任务的选择
elif dataset_type=="reg":
    Regtraintasks=["mu", "alpha",
         "homo", "gap", "r2", "zpve",
         "u0", "u298", "h298"]
    data_train = MyDatasetReg(remained_df, k_spt, k_query, Regtraintasks, Nway, epochs, type="train")
    dataset_train = data.DataLoader(dataset=data_train, batch_size=1, shuffle=True)
    Regtesttasks=["lumo","g298","cv"]
    data_test = MyDatasetReg(remained_df, k_spt, k_query, Regtesttasks, test_task_num, test_batchsz, type="test")
    dataset_test = data.DataLoader(dataset=data_test, batch_size=1, shuffle=True)
    

#
#
x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(
    [remained_smiles[0]], feature_dicts)
num_atom_features = x_atom.shape[-1]
num_bond_features = x_bonds.shape[-1]
# #
#创建多模态和Meta的联合模型，其中有一些参数是由SGGRL产生的，而训练方式采用了metalearner类
model = Fingerprint(radius, T, num_atom_features, num_bond_features,
                    fingerprint_dim, output_units_num, p_dropout, feature_dicts,
                    seq_data, data_3d, datas, seq_mask, args, device,compound_encoder_config).to(device)

#初始化了metalearner的一个实例
if dataset_type=="class":   
    meta = MetaLearner(model, device).to(device)

#添加归一化的回归任务
elif dataset_type=="reg":
    scaler_y = data_test.get_scaler()
    meta = MetaLearnerReg(model, device,scaler_y).to(device)


#
for epoch in range(episode):

    for step, (x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry) in enumerate(dataset_train):
        start_time = datetime.datetime.now()
        if dataset_type=="class": 
            accs_train, loss = meta(x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry)
            end_time = datetime.datetime.now()
            # print(" 元训练 耗时: {}秒".format(end_time - start_time))
            if step % 100 == 0:
                start_time = datetime.datetime.now()
                print(start_time, "    @@@@@@@@@@@@@@@@@@@@", flush=True)
                print("epoch:", epoch, "step:", step, flush=True)
                print(accs_train, flush=True)
                print(loss, flush=True)
        elif dataset_type=="reg":
            loss = meta(x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry)
            end_time = datetime.datetime.now()
            # print(" 元训练 耗时: {}秒".format(end_time - start_time))
            if step % 100 == 0:
                start_time = datetime.datetime.now()
                print(start_time, "    @@@@@@@@@@@@@@@@@@@@", flush=True)
                print("epoch:", epoch, "step:", step, flush=True)
                print(loss, flush=True)
        if metric=="accs":
            if step % 100 == 0:
                accs = [[] for i in range(test_task_num)]
                for x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry in dataset_test:
                    task_num = len(x_spt)
                    shot = len(x_spt[0])
                    query_size = len(x_qry[0])
                    # y_spt = y_spt.view(task_num, shot).long().cuda()
                    y_spt = y_spt.view(task_num, shot).long().to(device)
                    # y_qry = y_qry.view(task_num, query_size).long().cuda()
                    y_qry = y_qry.view(task_num, query_size).long().to(device)
                    for task_index, (x_spt_one, x_spt_idx_one, y_spt_one, x_qry_one, x_qry_idx_one, y_qry_one) in (
                            enumerate(zip(x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry))):
                        start_time = datetime.datetime.now()
                        test_acc = meta.finetunning_accs(x_spt_one,x_spt_idx_one, y_spt_one, x_qry_one, x_qry_idx_one, y_qry_one)
                        end_time = datetime.datetime.now()
                        # print(" 元测试 耗时: {}秒".format(end_time - start_time))
                        accs[task_index].append(test_acc)
                accs_res = np.array(accs).mean(axis=1).astype(np.float16)
                print('测试集准确率:', accs_res, flush=True)
        elif metric == "ROC":
            if step % 100 == 0:
                roc_aucs = [[] for _ in range(test_task_num)]
                for x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry in dataset_test:
                    task_num = len(x_spt)
                    shot = len(x_spt[0])
                    query_size = len(x_qry[0])
                    y_spt = y_spt.view(task_num, shot).long().to(device)
                    y_qry = y_qry.view(task_num, query_size).long().to(device)
                    for task_index, (x_spt_one, x_spt_idx_one, y_spt_one, x_qry_one, x_qry_idx_one, y_qry_one) in enumerate(zip(x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry)):
                        start_time = datetime.datetime.now()
                        roc_auc = meta.finetunning_ROC(x_spt_one, x_spt_idx_one, y_spt_one, x_qry_one, x_qry_idx_one, y_qry_one)
                        end_time = datetime.datetime.now()
                        roc_aucs[task_index].append(roc_auc)

                # 计算各个任务的平均 ROC-AUC
                roc_aucs_res = np.array(roc_aucs).mean(axis=1).astype(np.float16)

                # 计算所有任务的平均 ROC-AUC
                avg_roc_auc = roc_aucs_res.mean()

                print('测试集 各任务 ROC-AUC:', roc_aucs_res, flush=True)
                print('测试集 平均ROC-AUC:', avg_roc_auc, flush=True)
        elif metric=="MAE":
            if step % 100 == 0:
                maes = [[] for _ in range(test_task_num)]
                for x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry in dataset_test:
                    task_num = len(x_spt)
                    shot = len(x_spt[0])
                    query_size = len(x_qry[0])
                    y_spt = y_spt.view(task_num, shot).float().to(device)
                    y_qry = y_qry.view(task_num, query_size).float().to(device)
                    for task_index, (x_spt_one, x_spt_idx_one, y_spt_one, x_qry_one, x_qry_idx_one, y_qry_one) in enumerate(zip(x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry)):
                        start_time = datetime.datetime.now()
                        mae = meta.finetunning_MAE(x_spt_one, x_spt_idx_one, y_spt_one, x_qry_one, x_qry_idx_one, y_qry_one)
                        end_time = datetime.datetime.now()
                        maes[task_index].append(mae)
                maes_res = np.array(maes).mean(axis=1).astype(np.float16)
                print('测试集 MAE:', maes_res, flush=True)