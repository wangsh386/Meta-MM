

import torch.utils.data as data
import random
import numpy as np



class MyDataset(data.Dataset):
    def __init__(self, smiles_tasks_csv, k_shot, k_spt_pos, k_spt_neg, k_query, tasks, task_num, batchsz,
                 type):
        super(MyDataset, self).__init__()
        self.smiles_tasks_csv = smiles_tasks_csv

        self.k_shot = k_shot  # k-shot
        self.k_spt_pos = k_spt_pos  # k-shot positive
        self.k_spt_neg = k_spt_neg  # k-shot negative
        self.k_query = k_query  # for evaluation
        self.tasks = tasks
        self.task_num = task_num
        self.batchsz = batchsz


        self.all_smi = np.array(list(self.smiles_tasks_csv["remained_smiles"].values))
        # self.all_ori_smi = np.array(list(self.smiles_tasks_csv["smiles"].values))

        self.batchs_data = []
        self.create_batch2()

    def select_query(self, cls_label, index_list,):
        test_idx = np.random.choice(len(index_list), self.k_query, False)
        # np.random.shuffle(test_idx)
        query_x = list(self.all_smi[index_list[test_idx]])
        query_y = list(cls_label[index_list[test_idx]])

        # query_x = np.array(query_x)
        query_y = np.array(query_y)

        return query_x, query_y


    def create_batch2(self):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        # print('---------------create_batch222222222222222-------------------')
        for b in range(self.batchsz):  # for each batch
            # 1.select n_way classes randomly
            # selected_cls = random.choices(self.tasks, k=self.task_num)  #duplicate
            # 随机选取k个子任务，k=8
            selected_cls = random.sample(self.tasks, k=self.task_num)  # no duplicate
            # np.random.shuffle(selected_cls)     #####   2024/7/19
            support_x = []
            support_x_idx = []
            # support_ori_x = []
            support_y = []
            query_x = []
            # query_ori_x = []
            query_x_idx = []
            query_y = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                # 从df中取出对应列任务的数据的label
                cls_label = np.array(self.smiles_tasks_csv[cls])

                cls_support_x = []
                # cls_support_ori_x = []
                cls_support_idx_x = []
                cls_support_y = []
                cls_query_x = []
                # cls_query_ori_x = []
                cls_query_idx_x = []
                cls_query_y = []
                # label为0的index
                negative_index = np.where(cls_label == 0)[0]
                # k_spt_pos = 10  # k-shot positive
                # k_spt_neg = 10  # k-shot negative
                # k_query = 16  ## query data 的个数
                # 从整数范围内len(negative_index)随机选择size个，不放回，size=self.k_spt_neg + self.k_query //2
                # 随机抽样18个负样本
                negative_idx = np.random.choice(len(negative_index), self.k_spt_neg + self.k_query //2, False)
                np.random.shuffle(negative_idx)
                # 取出这18个的原始smiles、处理smiles和标签
                negative_all = list(self.all_smi[negative_index[negative_idx]])
                # negative_ori_all = list(self.all_ori_smi[negative_index[negative_idx]])
                negative_label_all = list(cls_label[negative_index[negative_idx]])

                # 前10个作为support
                # [794 738 202 772 408 682 532 857 956 919]
                negative_support_idx = negative_index[negative_idx][:self.k_spt_neg]
                cls_support_x.extend(negative_all[:self.k_spt_neg])
                # cls_support_ori_x.extend(negative_ori_all[:self.k_spt_neg])
                cls_support_idx_x.extend(negative_support_idx)
                cls_support_y.extend(negative_label_all[:self.k_spt_neg])

                # 后8个作为query
                # [1021  925 1046 1188  807  528  249  339]
                negative_query_idx = negative_index[negative_idx][self.k_spt_neg:]
                cls_query_x.extend(negative_all[self.k_spt_neg:])
                cls_query_idx_x.extend(negative_query_idx)
                # cls_query_ori_x.extend(negative_ori_all[self.k_spt_neg:])
                cls_query_y.extend(negative_label_all[self.k_spt_neg:])

                # 对于正样本，做同样的操作
                positive_index = np.where(cls_label == 1)[0]
                positive_idx = np.random.choice(len(positive_index), self.k_spt_pos + self.k_query //2, False)
                np.random.shuffle(positive_idx)

                positive_all = list(self.all_smi[positive_index[positive_idx]])
                # positive_ori_all = list(self.all_ori_smi[positive_index[positive_idx]])
                positive_label_all = list(cls_label[positive_index[positive_idx]])

                # [ 854 1245  697  250   42  474  482  920  453  895]
                positive_support_idx = positive_index[positive_idx][:self.k_spt_pos]
                cls_support_x.extend(positive_all[:self.k_spt_pos])
                cls_support_idx_x.extend(positive_support_idx)
                # cls_support_ori_x.extend(positive_ori_all[:self.k_spt_pos])
                cls_support_y.extend(positive_label_all[:self.k_spt_pos])

                # [ 171  768 1027  735 1332 1289  210 1338]
                positive_query_idx = positive_index[positive_idx][self.k_spt_pos:]
                cls_query_x.extend(positive_all[self.k_spt_pos:])
                # cls_query_ori_x.extend(positive_ori_all[self.k_spt_pos:])
                cls_query_idx_x.extend(positive_query_idx)
                cls_query_y.extend(positive_label_all[self.k_spt_pos:])

                # c = list(zip(cls_query_x,cls_query_ori_x, cls_query_y, query_idx))
                c = list(zip(cls_query_x, cls_query_y, cls_query_idx_x))
                random.shuffle(c)
                # cls_query_x[:], cls_query_ori_x[:], cls_query_y[:], query_idx[:] = zip(*c)
                cls_query_x[:], cls_query_y[:], cls_query_idx_x[:] = zip(*c)

                support_x.append(cls_support_x)
                # support_ori_x.append(cls_support_ori_x)
                support_x_idx.append(cls_support_idx_x)
                support_y.append(cls_support_y)

                query_x.append(cls_query_x)
                # query_ori_x.append(cls_query_ori_x)
                query_x_idx.append(cls_query_idx_x)
                query_y.append(cls_query_y)
            # support_x = np.array(support_x)
            support_y = np.array(support_y)


            # query_x = np.array(query_x)
            query_y = np.array(query_y)

            self.batchs_data.append([support_x, support_x_idx, support_y,
                                     query_x, query_x_idx, query_y])

    def __getitem__(self, item):
        x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry = self.batchs_data[item]
        return x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry

    def __len__(self):
        return len(self.batchs_data)