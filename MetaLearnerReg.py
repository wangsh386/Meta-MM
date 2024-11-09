import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score
# from ptflops import get_model_complexity_info
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
import time, datetime
from sklearn.metrics import roc_auc_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler

class MetaLearnerReg(nn.Module):
    def __init__(self, model,device,scaler_y):
        super(MetaLearnerReg, self).__init__()
        self.update_step = 5  ## task-level inner update steps
        self.update_step_test = 5
        self.net = model
        self.meta_lr = 0.001
        self.base_lr = 0.0001
        self.device = device
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.scaler_y = scaler_y
    #         self.meta_optim = torch.optim.SGD(self.net.parameters(), lr = self.meta_lr, momentum = 0.9, weight_decay=0.0005)

    def forward(self,x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry):
        # 初始化
        task_num = len(x_spt)
        shot = len(x_spt[0])
        query_size = len(x_qry[0])    #16
        # print(task_num)  #8 
        # print(shot)      #2
        # print(query_size)  #16
        loss_list_qry = [0 for _ in range(self.update_step + 1)]
        correct_list = [0 for _ in range(self.update_step + 1)]

        y_spt = y_spt.view(task_num, shot).float().to(self.device)
        y_qry = y_qry.view(task_num, query_size).float().to(self.device)

        for i in range(task_num):
            ## 第0步更新
            start_time = datetime.datetime.now()
            # print(list(self.net.parameters()))
            y_hat,x_list = self.net(x_spt[i], x_spt_idx[i], params=list(self.net.parameters()))  # (ways * shots, ways)
            y_hat = y_hat.to(self.device)
            end_time = datetime.datetime.now()
            # print(" 一次前向计算 耗时: {}秒".format(end_time - start_time))
            # flops, params = get_model_complexity_info(self.net, (x_spt[i], self.net.parameters()), as_strings=True, print_per_layer_stat=True)
            # print("%s |%s |%s" % ("mate_gat", flops, params))
            #
            # tensor = (x_spt[i], list(self.net.parameters()))
            # flops = FlopCountAnalysis(self.net, tensor)
            # print("FLOPs: ", flops.total())
            # print(f"y_hat shape1: {y_hat}, y_spt[{i}] shape1: {y_spt[i]}")
            # loss = F.mse_loss(y_hat, y_spt[i])
            
            all_loss, loss1, loss2 = self.net.loss_cal(x_list, y_hat, y_spt[i])
            loss = all_loss
            # grads = torch.autograd.grad(loss, list(self.net.parameters())[:38], create_graph=True, allow_unused=True)

            # # Ensure the gradient list has the same length as the parameters list
            # full_grads = list(grads) + [torch.zeros_like(p) for p in list(self.net.parameters())[38:]]

            # # Zip gradients and parameters together and update weights
            # tuples = zip(full_grads, self.net.parameters())
            # fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
            grad_main = torch.autograd.grad(loss, list(self.net.parameters())[:36])

            # 设置后面参数的梯度为0,因为多模态模型的params没有显式传播，因而在元学习部分暂不更新，在step部分再更新
            grad_multi_modal = [torch.zeros_like(p) for p in list(self.net.parameters())[36:]]
            
            full_grads = list(grad_main) + grad_multi_modal
            
            # grad = torch.autograd.grad(loss, self.net.parameters()) 
             
            # for idx, (g, p) in enumerate(zip(grad, self.net.parameters())):
            #     if g is None:
            #         print(f"Parameter {idx}: {p.shape}, Gradient is None")
            #     else:
            #         print(f"Parameter {idx}: {p.shape}, Gradient shape: {g.shape}")    
               
            tuples = zip(full_grads, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
            # 在query集上测试，计算准确率
            # 这一步使用更新前的数据
            with torch.no_grad():
                # aa1 = list(self.net.parameters())[0]
                y_hat,x_list = self.net(x_qry[i], x_qry_idx[i], list(self.net.parameters()))
                y_hat = y_hat.view(y_qry[i].shape).to(self.device)     #6.13
                # print(f"y_hat shape2: {y_hat}, y_qry[{i}] shape2: {y_qry[i]}")
                loss_qry, _, _ = self.net.loss_cal(x_list, y_hat, y_qry[i])
                loss_list_qry[0] += loss_qry.item()
                # pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
                # correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                # correct_list[0] += correct

            # 使用更新后的数据在query集上测试。
            with torch.no_grad():
                # aa2 = list(self.net.parameters())[0]
                # aa3 = list(fast_weights)[0]
                y_hat,x_list = self.net(x_qry[i], x_qry_idx[i], fast_weights)
                y_hat = y_hat.view(y_qry[i].shape).to(self.device)
                loss_qry, _, _ = self.net.loss_cal(x_list, y_hat, y_qry[i])
                loss_list_qry[1] += loss_qry.item()
                # pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
                # correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                # correct_list[1] += correct

            for k in range(1, self.update_step):
                # aa4 = list(self.net.parameters())[0]
                # aa5 = list(fast_weights)[0]
                y_hat,x_list = self.net(x_spt[i],  x_spt_idx[i], params=fast_weights)
                y_hat = y_hat.to(self.device)
                loss, _, _ = self.net.loss_cal(x_list, y_hat, y_spt[i])
                grad_main = torch.autograd.grad(loss, fast_weights[:36])

                # 设置后面参数的梯度为0,因为多模态模型的params没有显式传播，因而在元学习部分暂不更新，在step部分再更新
                grad_multi_modal = [torch.zeros_like(p) for p in fast_weights[36:]]
                
                full_grads = list(grad_main) + grad_multi_modal
                
                # loss = F.cross_entropy(y_hat, y_spt[i])
                # grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True, allow_unused=True)
                # fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, self.net.parameters())))
                # grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)
                
                # for idx, (g, p) in enumerate(zip(grad, fast_weights)):
                #     if g is None:
                #         print(f"Fast Weight {idx}: {p.shape}, Gradient is None")
                #     else:
                #         print(f"Fast Weight {idx}: {p.shape}, Gradient shape: {g.shape}")
                

                tuples = zip(full_grads, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))

                if k < self.update_step - 1:
                    with torch.no_grad():
                        y_hat,x_list = self.net(x_qry[i], x_qry_idx[i], fast_weights)
                        y_hat = y_hat.view(y_qry[i].shape).to(self.device)
                        loss_qry, _, _ = self.net.loss_cal(x_list, y_hat, y_qry[i])
                        loss_list_qry[k + 1] += loss_qry.item()
                else:
                    y_hat,x_list = self.net(x_qry[i], x_qry_idx[i], fast_weights)
                    y_hat = y_hat.view(y_qry[i].shape).to(self.device)
                    loss_qry, _, _ = self.net.loss_cal(x_list, y_hat, y_qry[i])
                    loss_list_qry[k + 1] += loss_qry

                # with torch.no_grad():
                #     pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
                #     correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                #     correct_list[k + 1] += correct
        #         print('hello')

        loss_qry = loss_list_qry[-1] / task_num
        self.meta_optim.zero_grad()  # 梯度清零
        loss_qry.backward()
        self.meta_optim.step()
        # aa6 = list(self.net.parameters())[0]
        # aa7 = list(fast_weights)[0]
        # accs = np.array(correct_list) / (query_size * task_num)
        loss_list_qry[-1] = loss_list_qry[-1].item()
        loss = np.array(loss_list_qry) / (task_num)
        return loss

    def finetunning_accs(self, x_spt, y_spt, x_qry, y_qry):

       query_size = len(x_qry)
       correct_list = [0 for _ in range(self.update_step_test + 1)]

       new_net = deepcopy(self.net)
       start_time = datetime.datetime.now()
       y_hat = new_net(x_spt, list(new_net.parameters()), "test_tranfer")
       end_time = datetime.datetime.now()
       # print(" 测试 一次前向计算 耗时: {}秒".format(end_time - start_time))
       loss = F.mse_loss(y_hat, y_spt)
       grad = torch.autograd.grad(loss, new_net.parameters())
       fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))

       # 在query集上测试，计算准确率
       # 这一步使用更新前的数据
       with torch.no_grad():
           y_hat = new_net(x_qry, list(new_net.parameters()), "test_tranfer")
           pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
           correct = torch.eq(pred_qry, y_qry).sum().item()
           correct_list[0] += correct


       # 使用更新后的数据在query集上测试。
       with torch.no_grad():
           y_hat = new_net(x_qry, fast_weights, "test_tranfer")
           pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
           correct = torch.eq(pred_qry, y_qry).sum().item()
           correct_list[1] += correct

       for k in range(1, self.update_step_test):
           y_hat = new_net(x_spt, fast_weights, "test_tranfer")
           loss = F.mse_loss(y_hat, y_spt)
           grad = torch.autograd.grad(loss, fast_weights)
           fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, fast_weights)))

           y_hat = new_net(x_qry, fast_weights, "test_tranfer")
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
    
    def finetunning_MAE(self, x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry):
        query_size = len(x_qry)
        y_qry_all = []
        y_pred_all = []

        new_net = deepcopy(self.net).to(self.device)
        y_hat,x_List = new_net(x_spt, x_spt_idx, list(new_net.parameters()), "test_transfer")
        loss, _, _ = self.net.loss_cal(x_list, y_hat, y_spt)
        grad_main = torch.autograd.grad(loss, list(new_net.parameters())[:36])

                # 设置后面参数的梯度为0,因为多模态模型的params没有显式传播，因而在元学习部分暂不更新，在step部分再更新
        grad_multi_modal = [torch.zeros_like(p) for p in list(new_net.parameters())[36:]]
                
        full_grads = list(grad_main) + grad_multi_modal
        
        # grad = torch.autograd.grad(loss, new_net.parameters())
        tuples = zip(full_grads, new_net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))

        with torch.no_grad():
            y_hat, x_list = new_net(x_qry, x_qry_idx, list(new_net.parameters()), "test_transfer")
            y_qry_all.append(y_qry.cpu().numpy())
            y_pred_all.append(y_hat.cpu().numpy())

        with torch.no_grad():
            y_hat, x_list = new_net(x_qry, x_qry_idx, fast_weights, "test_transfer")
            y_qry_all.append(y_qry.cpu().numpy())
            y_pred_all.append(y_hat.cpu().numpy())

        for k in range(1, self.update_step_test):
            y_hat, x_list = new_net(x_qry, x_qry_idx, list(new_net.parameters()), "test_transfer")
            loss, _, _ = self.net.loss_cal(x_list, y_hat, y_qry)
            grad = torch.autograd.grad(loss, new_net.parameters(), create_graph=True, retain_graph=True, allow_unused=True)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))

            y_hat, x_list = new_net(x_qry, x_qry_idx, fast_weights, "test_transfer")
            with torch.no_grad():
                y_qry_all.append(y_qry.cpu().numpy())
                y_pred_all.append(y_hat.cpu().numpy())

        del new_net
        # y_qry_all = np.concatenate(y_qry_all)
        # y_pred_all = np.concatenate(y_pred_all)
        y_qry_all = np.concatenate(y_qry_all).reshape(-1, 1)
        y_pred_all = np.concatenate(y_pred_all).reshape(-1, 1)
        # mae = mean_absolute_error(y_qry_all, y_pred_all)
        #反归一化
        mae = mean_absolute_error(self.scaler_y.inverse_transform(y_qry_all), self.scaler_y.inverse_transform(y_pred_all))
        # print("qry:",y_qry_all)
        # print("pred:",y_pred_all)
        # print(f"y_qry min: {y_qry_all.min()}, max: {y_qry_all.max()}, mean: {y_qry_all.mean()}")
        # print(f"y_hat min: {y_pred_all.min()}, max: {y_pred_all.max()}, mean: {y_pred_all.mean()}")
        return mae

    
    def finetunning_ROC(self, x_spt, y_spt, x_qry, y_qry):
        query_size = len(x_qry)
        y_qry_all = []
        y_pred_all = []

        new_net = deepcopy(self.net).to(self.device)
        y_hat = new_net(x_spt, list(new_net.parameters()), "test_transfer")
        loss = F.cross_entropy(y_hat, y_spt)
        grad = torch.autograd.grad(loss, new_net.parameters(), create_graph=True,retain_graph=True, allow_unused=True)
        fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))

        with torch.no_grad():
            y_hat= new_net(x_qry, list(new_net.parameters()), "test_transfer")
            y_qry_all.append(y_qry.cpu().numpy())
            y_pred_all.append(F.softmax(y_hat, dim=1)[:, 1].cpu().numpy())

        with torch.no_grad():
            y_hat = new_net(x_qry, fast_weights, "test_tranfer")
            y_qry_all.append(y_qry.cpu().numpy())
            y_pred_all.append(F.softmax(y_hat, dim=1)[:, 1].cpu().numpy())

        for k in range(1, self.update_step_test):
            y_hat = new_net(x_spt, list(new_net.parameters()), "test_tranfer")
            loss = F.cross_entropy(y_hat, y_spt)
            grad = torch.autograd.grad(loss, new_net.parameters(), create_graph=True,retain_graph=True, allow_unused=True)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))

            y_hat = new_net(x_qry, fast_weights, "test_tranfer")
            with torch.no_grad():
                y_qry_all.append(y_qry.cpu().numpy())
                y_pred_all.append(F.softmax(y_hat, dim=1)[:, 1].cpu().numpy())
        del new_net
        y_qry_all = np.concatenate(y_qry_all)
        y_pred_all = np.concatenate(y_pred_all)
        roc_auc = roc_auc_score(y_qry_all, y_pred_all)
        return roc_auc
        
