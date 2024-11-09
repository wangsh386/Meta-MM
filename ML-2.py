import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import time, datetime
torch.autograd.set_detect_anomaly(True)

class MetaLearner(nn.Module):
    def __init__(self, model, device):
        super(MetaLearner, self).__init__()
        self.update_step = 5  ## task-level inner update steps
        self.update_step_test = 5
        self.net = model
        self.meta_lr = 0.001
        self.base_lr = 0.0001
        self.device = device
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr, weight_decay=1e-5)

    #         self.meta_optim = torch.optim.SGD(self.net.parameters(), lr = self.meta_lr, momentum = 0.9, weight_decay=0.0005)


    def forward(self, x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry):
        # 初始化
        task_num = len(x_spt)
        shot = len(x_spt[0])
        query_size = len(x_qry[0])
        loss_list_qry = [0 for _ in range(self.update_step + 1)]
        correct_list = [0 for _ in range(self.update_step + 1)]

        # y_spt = y_spt.view(task_num, shot).long().cuda()
        y_spt = y_spt.view(task_num, shot).long().to(self.device)
        # y_qry = y_qry.view(task_num, query_size).long().cuda()
        y_qry = y_qry.view(task_num, query_size).long().to(self.device)
        for i in range(task_num):
            ## 第0步更新
            start_time = datetime.datetime.now()

            y_hat = self.net(x_spt[i], x_spt_idx[i], params=list(self.net.parameters()))  # (ways * shots, ways)
            end_time = datetime.datetime.now()
            # print(" 一次前向计算 耗时: {}秒".format(end_time - start_time))
            # flops, params = get_model_complexity_info(self.net, (x_spt[i], self.net.parameters()), as_strings=True, print_per_layer_stat=True)
            # print("%s |%s |%s" % ("mate_gat", flops, params))
            #
            # tensor = (x_spt[i], list(self.net.parameters()))
            # flops = FlopCountAnalysis(self.net, tensor)
            # print("FLOPs: ", flops.total())

            loss = F.cross_entropy(y_hat, y_spt[i])

            # print(list(self.net.parameters())) # 393  # 38  # 353
            grad = torch.autograd.grad(loss, self.net.parameters(), allow_unused=True)
            # aa = list(self.net.parameters())[0]
            tuples = zip(list(grad)[:38], list(self.net.parameters())[:38])  ## 将梯度和参数\theta一一对应起来
            # print(list(grad))
            # print('------------------')
            # print(list(self.net.parameters()))
            # fast_weights这一步相当于求了一个\theta - \alpha*\nabla(L)
            # print()
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
            fast_weights = fast_weights + list(self.net.parameters())[38:]
            # 在query集上测试，计算准确率
            # 这一步使用更新前的数据
            with torch.no_grad():
                # aa1 = list(self.net.parameters())[0]
                y_hat = self.net(x_qry[i], x_qry_idx[i], list(self.net.parameters()))
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[0] += loss_qry.item()
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
                correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                correct_list[0] += correct

            # 使用更新后的数据在query集上测试。
            with torch.no_grad():
                # aa2 = list(self.net.parameters())[0]
                # aa3 = list(fast_weights)[0]
                y_hat = self.net(x_qry[i], x_qry_idx[i], fast_weights)
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[1] += loss_qry.item()
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
                correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                correct_list[1] += correct

            for k in range(1, self.update_step):
                # aa4 = list(self.net.parameters())[0]
                # aa5 = list(fast_weights)[0]
                y_hat = self.net(x_spt[i],  x_spt_idx[i], params=fast_weights)
                loss = F.cross_entropy(y_hat, y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)
                tuples = zip(list(grad)[:38], list(fast_weights)[:38])

                fast_weights_tmp = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))

                fast_weights = fast_weights_tmp + list(fast_weights)[38:]

                if k < self.update_step - 1:
                    with torch.no_grad():
                        y_hat = self.net(x_qry[i],x_qry_idx[i],  params=fast_weights)
                        loss_qry = F.cross_entropy(y_hat, y_qry[i])
                        loss_list_qry[k + 1] += loss_qry.item()
                else:
                    y_hat = self.net(x_qry[i], x_qry_idx[i], params=fast_weights)
                    loss_qry = F.cross_entropy(y_hat, y_qry[i])
                    loss_list_qry[k + 1] += loss_qry

                with torch.no_grad():
                    pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                    correct_list[k + 1] += correct
        #         print('hello')

        loss_qry = loss_list_qry[-1] / task_num
        self.meta_optim.zero_grad()  # 梯度清零
        loss_qry.backward()
        self.meta_optim.step()
        # aa6 = list(self.net.parameters())[0]
        # aa7 = list(fast_weights)[0]
        accs = np.array(correct_list) / (query_size * task_num)
        loss_list_qry[-1] = loss_list_qry[-1].item()
        loss = np.array(loss_list_qry) / (task_num)
        return accs, loss

    def finetunning(self, x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry):

        query_size = len(x_qry)
        correct_list = [0 for _ in range(self.update_step_test + 1)]

        new_net = deepcopy(self.net)
        start_time = datetime.datetime.now()
        y_hat = new_net(x_spt, x_spt_idx, list(new_net.parameters()), "test_tranfer")
        end_time = datetime.datetime.now()
        # print(" 测试 一次前向计算 耗时: {}秒".format(end_time - start_time))
        loss = F.cross_entropy(y_hat, y_spt)
        grad = torch.autograd.grad(loss, new_net.parameters(), allow_unused=True)
        fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0],
                                zip(list(grad)[:38], list(new_net.parameters())[:38])))
        fast_weights = fast_weights + list(new_net.parameters())[38:]

        # 在query集上测试，计算准确率
        # 这一步使用更新前的数据
        with torch.no_grad():
            y_hat = new_net(x_qry, x_qry_idx, list(new_net.parameters()), "test_tranfer")
            pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
            correct = torch.eq(pred_qry, y_qry).sum().item()
            correct_list[0] += correct


        # 使用更新后的数据在query集上测试。
        with torch.no_grad():
            y_hat = new_net(x_qry, x_qry_idx, fast_weights, "test_tranfer")
            pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
            correct = torch.eq(pred_qry, y_qry).sum().item()
            correct_list[1] += correct

        for k in range(1, self.update_step_test):
            y_hat = new_net(x_spt, x_spt_idx, fast_weights, "test_tranfer")
            loss = F.cross_entropy(y_hat, y_spt)

            grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)

            fast_weights_tmp = list(map(lambda p: p[1] - self.base_lr * p[0],
                                    zip(list(grad)[:38], list(fast_weights)[:38])))
            fast_weights = fast_weights_tmp + list(fast_weights)[38:]

            y_hat = new_net(x_qry, x_qry_idx, fast_weights, "test_tranfer")
            with torch.no_grad():
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
                correct = torch.eq(pred_qry, y_qry).sum().item()
                correct_list[k + 1] += correct

                # w, p = wilcoxon(pred_qry.cpu().numpy(), y_qry.cpu().numpy())
                # print("wilcoxon", w, p)
                # print("kappa", cohen_kappa_score(pred_qry.cpu().numpy(), y_qry.cpu().numpy()))

        del new_net
        accs = np.array(correct_list) / query_size
        return accs

    # def finetunning_all(self, x_spt, y_spt, x_qry, y_qry):
    #
    #     query_size = len(x_qry)
    #     correct_list = [0 for _ in range(self.update_step_test + 1)]
    #
    #     new_net = deepcopy(self.net)
    #     y_hat = new_net(x_spt, list(new_net.parameters()))
    #     loss = F.cross_entropy(y_hat, y_spt)
    #     grad = torch.autograd.grad(loss, new_net.parameters())
    #     fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))
    #
    #     # 在query集上测试，计算准确率
    #     # 这一步使用更新前的数据
    #     with torch.no_grad():
    #         y_hat = new_net(x_qry, list(new_net.parameters()))
    #         pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
    #         correct = torch.eq(pred_qry, y_qry).sum().item()
    #         correct_list[0] += correct
    #
    #     # 使用更新后的数据在query集上测试。
    #     with torch.no_grad():
    #         y_hat = new_net(x_qry, params=fast_weights)
    #         pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
    #         correct = torch.eq(pred_qry, y_qry).sum().item()
    #         correct_list[1] += correct
    #
    #     for k in range(1, self.update_step_test):
    #         y_hat = new_net(x_spt, params=fast_weights)
    #         loss = F.cross_entropy(y_hat, y_spt)
    #         grad = torch.autograd.grad(loss, fast_weights)
    #         fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, fast_weights)))
    #
    #         y_hat = new_net(x_qry, fast_weights)
    #
    #         with torch.no_grad():
    #             pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
    #             correct = torch.eq(pred_qry, y_qry).sum().item()
    #             correct_list[k + 1] += correct
    #
    #     del new_net
    #     accs = np.array(correct_list) / query_size
    #     return accs
