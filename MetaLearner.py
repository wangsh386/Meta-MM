import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import time, datetime
from sklearn.metrics import roc_auc_score


class MetaLearner(nn.Module):
    def __init__(self, model, device):
        super(MetaLearner, self).__init__()
        self.update_step = 5  ## task-level inner update steps
        self.update_step_test = 5
        self.net = model
        self.meta_lr = 0.001
        self.base_lr = 0.0001
        self.device = device
        #adam型优化器，parameter即全部的参数
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)# weight_decay=1e-5

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
        #
        for i in range(task_num):
            # print("task_num",i)
            ## 第0步更新
            start_time = datetime.datetime.now()
            #调用了model里的forward函数进行一次预测
            y_hat,x_list = self.net(x_spt[i], x_spt_idx[i], params=list(self.net.parameters()))  # (ways * shots, ways)
            end_time = datetime.datetime.now()           
            #使用交叉熵定义损失
#            loss = F.cross_entropy(y_hat, y_spt[i])

            #使用多模态误差定义损失，去掉mask
            #对于回归任务，需要在args的loss_type进行修改
            ###
            all_loss, loss1, loss2 = self.net.loss_cal(x_list, y_hat, y_spt[i])
            
            loss = all_loss
            # loss = F.cross_entropy(y_hat, y_spt[i])
            # ###
            # 计算前36个参数的梯度
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
            # print("self.net.parameters():")
            # for idx, param in enumerate(self.net.parameters()):
            #     print(f"Param {idx}: Shape {param.shape}, Requires Grad {param.requires_grad}")

            # # 打印 fast_weights 的情况
            # print("\nfast_weights:")
            # for idx, param in enumerate(fast_weights):
            #     print(f"Fast Weight {idx}: Shape {param.shape}, Requires Grad {param.requires_grad}")
            # Calculate gradients for the first 38 parameters
            # print(list(self.net.parameters())[:38])
            # grads = torch.autograd.grad(loss, list(self.net.parameters())[:38])
            # # print("grad",list(grads))
            # # Ensure the gradient list has the same length as the parameters list
            # full_grads = list(grads) + [torch.zeros_like(p) for p in list(self.net.parameters())[38:]]
            # # print("full_grad",list(full_grads))
            # # Zip gradients and parameters together and update weights
            # tuples = zip(full_grads, self.net.parameters())
            # fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))

            #fast_weights这一步相当于求了一个\theta - \alpha*\nabla(L)
            #print()
            #fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
            # 在query集上测试，计算准确率
            # 这一步使用更新前的数据
            
            #y_hat实质上是四个的结合,xlist是没有经过全连接层的四个特征
            with torch.no_grad():
                # aa1 = list(self.net.parameters())[0]
                y_hat,x_list = self.net(x_qry[i], x_qry_idx[i], list(self.net.parameters()))
                # loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_qry, _, _ = self.net.loss_cal(x_list, y_hat, y_qry[i])
                loss_list_qry[0] += loss_qry.item()
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
                correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                correct_list[0] += correct

            # loss.backward()
            # self.meta_optim.step()
            
            # 使用更新后的数据在query集上测试。
            with torch.no_grad():
                # aa2 = list(self.net.parameters())[0]
                # aa3 = list(fast_weights)[0]
                # y_hat = self.net(x_qry[i], x_qry_idx[i], fast_weights)
                y_hat,x_list = self.net(x_qry[i], x_qry_idx[i], fast_weights)
                # loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_qry, _, _ = self.net.loss_cal(x_list, y_hat, y_qry[i])
                loss_list_qry[1] += loss_qry.item()
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
                correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                correct_list[1] += correct

            for k in range(1, self.update_step):
                # print("update_step",k)
                # aa4 = list(self.net.parameters())[0]
                # aa5 = list(fast_weights)[0]
                y_hat,x_list = self.net(x_spt[i],  x_spt_idx[i], params=fast_weights)
                
                all_loss, loss1, loss2 = self.net.loss_cal(x_list, y_hat, y_spt[i])
                loss=all_loss
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
                # grads = torch.autograd.grad(loss, list(self.net.parameters())[:38])

                # # Ensure the gradient list has the same length as the parameters list
                # full_grads = list(grads) + [torch.zeros_like(p) for p in list(self.net.parameters())[38:]]

                # # Zip gradients and parameters together and update weights
                # tuples = zip(full_grads, fast_weights)
                # fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
                if k < self.update_step - 1:
                    with torch.no_grad():
                        # y_hat = self.net(x_qry[i],x_qry_idx[i],  params=fast_weights)
                        y_hat,x_list = self.net(x_qry[i],x_qry_idx[i], params=fast_weights)
                        # y_hat = self.net(x_qry[i],x_qry_idx[i],  params=list(self.net.parameters())).detach()
                        # loss_qry = F.cross_entropy(y_hat, y_qry[i])
                        loss_qry, _, _ = self.net.loss_cal(x_list, y_hat, y_qry[i])
                        loss_list_qry[k + 1] += loss_qry.item()
                else:
                    # y_hat = self.net(x_qry[i], x_qry_idx[i], params=fast_weights)
                    y_hat,x_list = self.net(x_qry[i], x_qry_idx[i], params=fast_weights)
                    # y_hat = self.net(x_qry[i], x_qry_idx[i], params=list(self.net.parameters())).detach()
                    # loss_qry = F.cross_entropy(y_hat, y_qry[i])
                    loss_qry, _, _ = self.net.loss_cal(x_list, y_hat, y_qry[i])
                    loss_list_qry[k + 1] += loss_qry

                with torch.no_grad():
                    pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                    correct_list[k + 1] += correct
        
        # print("loss_qry",loss_list_qry)
        loss_qry = loss_list_qry[-1] / task_num  #, device=self.device, requires_grad=True
        # print("loss_qry",loss_qry)
        self.meta_optim.zero_grad()  # 梯度清零
        
        loss_qry.backward()
        # for idx, param in enumerate(self.net.parameters()):
        #     if param.grad is None:
        #         print(f"Parameter {idx}: {param.shape}, Gradient is None")
        #     else:
        #         if torch.all(param.grad == 0):
        #             print(f"Parameter {idx}: {param.shape}, Gradient is zero")
        #         else:
        #             print(f"Parameter {idx}: {param.shape}, Gradient shape: {param.grad.shape}, Gradient: {param.grad}")

        self.meta_optim.step()
#        # aa6 = list(self.net.parameters())[0]
#        # aa7 = list(fast_weights)[0]
        accs = np.array(correct_list) / (query_size * task_num)
        # loss_list_qry = [loss.cpu().item() for loss in loss_list_qry]  # 将所有损失从GPU转移到CPU
        loss_list_qry[-1] = loss_list_qry[-1].item()
        loss = np.array(loss_list_qry) / (task_num)
        return accs, loss
#返回正确率
    def finetunning_accs(self, x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry):
        query_size = len(x_qry)
        y_qry_all = []
        y_pred_all = []

        new_net = deepcopy(self.net).to(self.device)
        # 初始化 fast_weights
        y_hat, x_list = new_net(x_spt, x_spt_idx, list(new_net.parameters()), "test_transfer")
        all_loss, loss1, loss2 = new_net.loss_cal(x_list, y_hat, y_spt)
        loss = all_loss
        grads = torch.autograd.grad(loss, list(self.net.parameters())[:38], create_graph=True, allow_unused=True)

            # Ensure the gradient list has the same length as the parameters list
        full_grads = list(grads) + [torch.zeros_like(p) for p in list(self.net.parameters())[38:]]

            # Zip gradients and parameters together and update weights
        tuples = zip(full_grads, self.net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
        # grad = torch.autograd.grad(loss, new_net.parameters(), create_graph=True,retain_graph=True, allow_unused=True)
        # fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))

        # 第一次评估
        with torch.no_grad():
            y_hat, x_list = new_net(x_qry, x_qry_idx, list(new_net.parameters()), "test_transfer")
            y_qry_all.append(y_qry.cpu().numpy())
            y_pred_all.append(F.softmax(y_hat, dim=1)[:, 1].cpu().numpy())

        # 使用 fast_weights 进行评估
        with torch.no_grad():
            y_hat, x_list = new_net(x_qry, x_qry_idx, fast_weights, "test_tranfer")
            y_qry_all.append(y_qry.cpu().numpy())
            y_pred_all.append(F.softmax(y_hat, dim=1)[:, 1].cpu().numpy())

        for k in range(1, self.update_step_test):
            y_hat, x_list = new_net(x_spt, x_spt_idx, fast_weights, "test_transfer")
            all_loss, loss1, loss2 = new_net.loss_cal(x_list, y_hat, y_spt)
            loss = all_loss
            grads = torch.autograd.grad(loss, list(self.net.parameters())[:38], create_graph=True, allow_unused=True)

            full_grads = list(grads) + [torch.zeros_like(p) for p in list(self.net.parameters())[38:]]

            tuples = zip(full_grads, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))

            y_hat, x_list = new_net(x_qry, x_qry_idx, fast_weights, "test_tranfer")
            with torch.no_grad():
                y_qry_all.append(y_qry.cpu().numpy())
                y_pred_all.append(F.softmax(y_hat, dim=1)[:, 1].cpu().numpy())

        del new_net
        y_qry_all = np.concatenate(y_qry_all)
        y_pred_all = np.concatenate(y_pred_all)
        roc_auc = roc_auc_score(y_qry_all, y_pred_all)
        return roc_auc
        
    def finetunning_ROC(self, x_spt, x_spt_idx, y_spt, x_qry, x_qry_idx, y_qry):
        query_size = len(x_qry)
        y_qry_all = []
        y_pred_all = []

        new_net = deepcopy(self.net).to(self.device)
        #support
        y_hat,x_list = new_net(x_spt, x_spt_idx, list(new_net.parameters()), "test_transfer")
        loss, loss1, loss2 = new_net.loss_cal(x_list, y_hat, y_spt)
        grad_main = torch.autograd.grad(loss, list(new_net.parameters())[:36])

                # 设置后面参数的梯度为0,因为多模态模型的params没有显式传播，因而在元学习部分暂不更新，在step部分再更新
        grad_multi_modal = [torch.zeros_like(p) for p in list(new_net.parameters())[36:]]
                
        full_grads = list(grad_main) + grad_multi_modal
        
        # grad = torch.autograd.grad(loss, new_net.parameters())
        tuples = zip(full_grads, new_net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))

        with torch.no_grad():
           y_hat,x_list= new_net(x_qry, x_qry_idx, list(new_net.parameters()), "test_transfer")
           y_qry_all.append(y_qry.cpu().numpy())
           y_pred_all.append(F.softmax(y_hat, dim=1)[:, 1].cpu().numpy())

        with torch.no_grad():
           y_hat,x_list = new_net(x_qry, x_qry_idx, fast_weights, "test_tranfer")
           y_qry_all.append(y_qry.cpu().numpy())
           y_pred_all.append(F.softmax(y_hat, dim=1)[:, 1].cpu().numpy())

        for k in range(1, self.update_step_test):
        #    print(f"y_hat shape2: {y_hat}, y_qry shape2: {y_qry}")
           #query:16 and 20 not match
            y_hat,x_list = new_net(x_spt, x_spt_idx, fast_weights, "test_tranfer")
            loss, loss1, loss2 = new_net.loss_cal(x_list, y_hat, y_spt)
            
            # grad = torch.autograd.grad(loss, new_net.parameters())
            
            grad_main = torch.autograd.grad(loss, fast_weights[:36])

                # 设置后面参数的梯度为0,因为多模态模型的params没有显式传播，因而在元学习部分暂不更新，在step部分再更新
            grad_multi_modal = [torch.zeros_like(p) for p in fast_weights[36:]]
                    
            full_grads = list(grad_main) + grad_multi_modal
            tuples = zip(full_grads, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
            

            y_hat,x_list = new_net(x_qry, x_qry_idx, fast_weights, "test_tranfer")
            with torch.no_grad():
               y_qry_all.append(y_qry.cpu().numpy())
               y_pred_all.append(F.softmax(y_hat, dim=1)[:, 1].cpu().numpy())
        del new_net
        y_qry_all = np.concatenate(y_qry_all)
        y_pred_all = np.concatenate(y_pred_all)
        roc_auc = roc_auc_score(y_qry_all, y_pred_all)
        return roc_auc


