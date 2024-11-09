import torch.utils.data as data
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
class MyDatasetReg(data.Dataset):  # 定义MyDataset类，继承自data.Dataset
    def __init__(self, smiles_tasks_csv, k_shot, k_query, tasks, task_num, batchsz, type):
        super(MyDatasetReg, self).__init__()  # 调用父类的构造函数
        self.smiles_tasks_csv = smiles_tasks_csv  # 保存化学数据集，包含SMILES字符串和任务标签
        self.k_shot = k_shot  # 保存每个类别的样本数
        self.k_query = k_query  # 保存查询集的样本数
        self.tasks = tasks  # 保存任务列表
        self.task_num = task_num  # 保存任务数量
        self.batchsz = batchsz  # 保存批次大小
        self.all_smi = np.array(list(self.smiles_tasks_csv["remained_smiles"].values))  # 将所有SMILES字符串转换为NumPy数组
        self.batchs_data = []  # 初始化批次数据列表
        self.create_batch2()  # 调用create_batch2方法创建批次数据
        self.scaler_y = StandardScaler()
        for task in tasks:
            self.smiles_tasks_csv[task] = self.scaler_y.fit_transform(self.smiles_tasks_csv[task].values.reshape(-1, 1))

    def select_query(self, index_list):
        test_idx = np.random.choice(len(index_list), self.k_query, False)  # 随机选择查询集的索引
        np.random.shuffle(test_idx)  # 打乱索引
        query_x = list(self.all_smi[index_list[test_idx]])  # 查询集的SMILES字符串
        query_y = list(self.smiles_tasks_csv.iloc[index_list[test_idx], 1].values)  # 查询集的标签
        query_y = np.array(query_y)  # 将查询集标签转换为NumPy数组
        return query_x, query_y  # 返回查询集的SMILES字符串和标签

    def create_batch2(self):  # 定义create_batch2方法，用于创建批次数据
        for b in range(self.batchsz):  # 遍历每个批次
            selected_cls = random.sample(self.tasks, k=self.task_num)  # 随机选择无重复的任务类别
            np.random.shuffle(selected_cls)  # 打乱选中的任务类别
            support_x = []  # 初始化支持集的SMILES字符串列表
            support_x_idx = []  # 初始化支持集的索引列表
            support_y = []  # 初始化支持集的标签列表
            query_x = []  # 初始化查询集的SMILES字符串列表
            query_x_idx = []  # 初始化查询集的索引列表
            query_y = []  # 初始化查询集的标签列表
            for cls in selected_cls:  # 遍历选中的任务类别
                cls_label = np.array(self.smiles_tasks_csv[cls])  # 获取当前类别的标签数组
                cls_support_x = []  # 初始化当前类别支持集的SMILES字符串列表
                cls_support_idx_x = []  # 初始化当前类别支持集的索引列表
                cls_support_y = []  # 初始化当前类别支持集的标签列表
                cls_query_x = []  # 初始化当前类别查询集的SMILES字符串列表
                cls_query_idx_x = []  # 初始化当前类别查询集的索引列表
                cls_query_y = []  # 初始化当前类别查询集的标签列表

                # 从所有样本中随机选择支持集和查询集
                index_list = np.arange(len(cls_label))  # 获取所有样本的索引
                np.random.shuffle(index_list)  # 打乱索引

                support_idx = index_list[:self.k_shot]  # 选择前k_shot个样本作为支持集
                query_idx = index_list[self.k_shot:self.k_shot + self.k_query]  # 选择接下来的k_query个样本作为查询集

                cls_support_x = list(self.all_smi[support_idx])  # 获取支持集的SMILES字符串
                cls_support_idx_x = list(support_idx)  # 获取支持集的索引
                cls_support_y = list(self.smiles_tasks_csv.iloc[support_idx, 1].values)  # 获取支持集的标签
                cls_query_x = list(self.all_smi[query_idx])  # 获取查询集的SMILES字符串
                cls_query_idx_x = list(query_idx)  # 获取查询集的索引
                cls_query_y = list(self.smiles_tasks_csv.iloc[query_idx, 1].values)  # 获取查询集的标签

                support_x.append(cls_support_x)  # 将当前类别的支持集SMILES字符串添加到总支持集列表
                support_x_idx.append(cls_support_idx_x)  # 将当前类别的支持集索引添加到总支持集列表
                support_y.append(cls_support_y)  # 将当前类别的支持集标签添加到总支持集列表
                query_x.append(cls_query_x)  # 将当前类别的查询集SMILES字符串添加到总查询集列表
                query_x_idx.append(cls_query_idx_x)  # 将当前类别的查询集索引添加到总查询集列表
                query_y.append(cls_query_y)  # 将当前类别的查询集标签添加到总查询集列表

            support_y = np.array(support_y)  # 将支持集标签列表转换为NumPy数组
            query_y = np.array(query_y)  # 将查询集标签列表转换为NumPy数组
            self.batchs_data.append([support_x, support_x_idx, support_y, query_x, query_x_idx, query_y])  # 将当前批次的数据添加到批次数据列表

    def __getitem__(self, item):  # 定义获取数据项的方法
        x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry = self.batchs_data[item]  # 获取指定索引的批次数据
        return x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry  # 返回支持集和查询集的SMILES字符串和标签

    def __len__(self):  # 定义获取数据集长度的方法
        return len(self.batchs_data)  # 返回批次数据列表的长度
    def get_scaler(self):
        return self.scaler_y