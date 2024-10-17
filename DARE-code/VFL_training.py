"""
@author: tzh666
@context: VFL training
"""
import logging
logging.basicConfig(level=logging.INFO, filename="mylog.log", filemode='w', format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import os, time, json, random, dill, copy
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.VFLModel import VFLBottomModel, VFLTopModel
from utils import set_loaders, split_data, perturb_representation
from params import params_init


class VFLModel(nn.Module):

    def __init__(self):
        super(VFLModel, self).__init__()
        ## loss setting
        self.loss_func_top_model = nn.CrossEntropyLoss()
        self.loss_func_bottom_model = keep_predict_loss
        ## model setting
        self.bottom_model_a = VFLBottomModel(args)
        self.bottom_model_b = VFLBottomModel(args)
        self.top_model = VFLTopModel(args)
        ## optimizer setting
        self.optimizer_top_model = optim.SGD(self.top_model.parameters(), lr=args.vfl_lr,
                                             momentum=args.vfl_momentum, weight_decay=args.vfl_weight_decay)
        self.optimizer_bottom_model_a = optim.SGD(self.bottom_model_a.parameters(), lr=args.vfl_lr,
                                                  momentum=args.vfl_momentum, weight_decay=args.vfl_weight_decay)
        self.optimizer_bottom_model_b = optim.SGD(self.bottom_model_b.parameters(), lr=args.vfl_lr,
                                                  momentum=args.vfl_momentum, weight_decay=args.vfl_weight_decay)

    def forward(self, x):
        # in vertical federated setting, each party has non-lapping features of the same sample
        x_a, x_b = split_data(args, x)
        out_a = self.bottom_model_a(x_a)
        out_b = self.bottom_model_b(x_b)
        out = self.top_model(out_a, out_b)

        return out

    # training one epoch of a batch
    # bottom models forward, top model forward, top model backward and update, bottom backward and update
    def train_round_per_batch(self, data, target):
        # store grad of input of top model/outputs of bottom models
        input_top_model_a = torch.tensor([], requires_grad=True)
        input_top_model_b = torch.tensor([], requires_grad=True)

        ## data setting
        x_a, x_b = split_data(args, data)

        ### model setting
        ## bottom model - forward propagation
        self.bottom_model_a.train(mode=True)
        output_bottom_model_a = self.bottom_model_a(x_a)
        self.bottom_model_b.train(mode=True)
        output_bottom_model_b = self.bottom_model_b(x_b)

        ## top model - forward propagation
        # by concatenating output of bottom a/b(dim=10+10=20), we get input of top model
        input_top_model_a.data = output_bottom_model_a.data
        input_top_model_b.data = output_bottom_model_b.data
        self.top_model.train(mode=True)
        output_framework = self.top_model(input_top_model_a, input_top_model_b)

        ## top model - backward propagation
        # update
        loss_framework = update_top_model_one_batch(optimizer=self.optimizer_top_model,
                                                    model=self.top_model,
                                                    output=output_framework,
                                                    batch_target=target,
                                                    loss_func=self.loss_func_top_model)
        # acquire grad
        # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
        grad_output_bottom_model_a = input_top_model_a.grad
        grad_output_bottom_model_b = input_top_model_b.grad

        model_all_layers_grads_list = [grad_output_bottom_model_a, grad_output_bottom_model_b]
        grad_output_bottom_model_a, grad_output_bottom_model_b = tuple(model_all_layers_grads_list)

        ## bottom model - backward propagation
        # update
        update_bottom_model_one_batch(optimizer=self.optimizer_bottom_model_a,
                                      model=self.bottom_model_a,
                                      output=output_bottom_model_a,
                                      batch_target=grad_output_bottom_model_a,
                                      loss_func=self.loss_func_bottom_model,
                                      data=x_a)

        update_bottom_model_one_batch(optimizer=self.optimizer_bottom_model_b,
                                      model=self.bottom_model_b,
                                      output=output_bottom_model_b,
                                      batch_target=grad_output_bottom_model_b,
                                      loss_func=self.loss_func_bottom_model,
                                      data=x_b)

        return loss_framework


# main function of VFL training 
def VFL_train_main(args):

    model = VFLModel()
    model = model.to(args.device)

    # optimizer setting
    stone1 = int(args.vfl_epochs * 0.5)   # Specify location for learning rate decay
    stone2 = int(args.vfl_epochs * 0.8)
    lr_scheduler_top_model = torch.optim.lr_scheduler.MultiStepLR(model.optimizer_top_model,
                                                                  milestones=[stone1, stone2], gamma=args.vfl_step_gamma)
    lr_scheduler_model_a = torch.optim.lr_scheduler.MultiStepLR(model.optimizer_bottom_model_a,
                                                                milestones=[stone1, stone2], gamma=args.vfl_step_gamma)
    lr_scheduler_model_b = torch.optim.lr_scheduler.MultiStepLR(model.optimizer_bottom_model_b,
                                                                milestones=[stone1, stone2], gamma=args.vfl_step_gamma)
    # loader setting
    train_loader, val_loader, _ = set_loaders(args)

    # start training
    print("VFL Training begin!")
    for epoch in range(args.vfl_epochs):
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float().to(args.device)
            target = target.long().to(args.device)
            # train one epoch of a batch
            loss_framework = model.train_round_per_batch(data, target)
            # print training loss
            if batch_idx % 25 == 0:
                num_samples = len(train_loader.dataset)
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), num_samples,
                           100. * batch_idx / len(train_loader), loss_framework.data.item()))

        lr_scheduler_top_model.step()
        lr_scheduler_model_a.step()
        lr_scheduler_model_b.step()

        print('-----------------VFL Evaluation-----------------------')
        print('Epoch:{}, time:{}'.format(epoch, time.time()-start))
        print('Evaluation on the training dataset:')
        test_per_epoch(args, test_loader=train_loader, model=model, loss_func_top_model=model.loss_func_top_model)
        print('Evaluation on the testing dataset:')
        test_per_epoch(args, test_loader=val_loader, model=model, loss_func_top_model=model.loss_func_top_model)

    # Save the model. The model has lambda layers, so dill must be used.
    if args.save:
        torch.save(model.state_dict(), f"{args.vfl_dir_save_model}/{args.vfl_epochs}epochs_2users.pth", pickle_module=dill)
        torch.save(model.bottom_model_a.state_dict(), f"{args.vfl_dir_save_model}/{args.vfl_epochs}epochs_2users_bottom.pth", pickle_module=dill)


def test_per_epoch(args, test_loader, model, loss_func_top_model=None, k=5):
    if args.dataset == 'Bank':
        k = 1
    test_loss = 0
    correct_top1 = 0
    correct_topk = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.float().to(args.device)
            target = target.long().to(args.device)

            # set all sub-models to eval mode.
            model.bottom_model_a.eval()
            model.bottom_model_b.eval()
            model.top_model.eval()

            # run forward process of the whole framework
            x_a, x_b = split_data(args, data)
            output_bottom_model_a = model.bottom_model_a(x_a)
            output_bottom_model_b = model.bottom_model_b(x_b)

            output_model = model.top_model(output_bottom_model_a, output_bottom_model_b)

            correct_top1_batch, correct_topk_batch = correct_counter(output_model, target, (1, k))

            # sum up batch loss
            test_loss += loss_func_top_model(output_model, target).data.item()

            correct_top1 += correct_top1_batch
            correct_topk += correct_topk_batch

        num_samples = len(test_loader.dataset)
        test_loss /= num_samples
        print('Loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%), Top {} Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct_top1, num_samples, 100.00 * float(correct_top1) / num_samples, k,
            correct_topk, num_samples, 100.00 * float(correct_topk) / num_samples))


def keep_predict_loss(y_true, y_pred):
    return torch.sum(y_true * y_pred)


def correct_counter(output, target, topk=(1, 5)):
    correct_counts = []
    for k in topk:
        _, pred = output.topk(k, 1, True, True)
        correct_k = torch.eq(pred, target.view(-1, 1)).sum().float().item()
        correct_counts.append(correct_k)
    return correct_counts


def update_top_model_one_batch(optimizer, model, output, batch_target, loss_func):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def update_bottom_model_one_batch(optimizer, model, output, batch_target, loss_func, data):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()

    # add dp or mask
    if args.attack_noise_type == "Soteria":
        param = list(model.net.linear.parameters())[0]
        gradient = perturb_representation(param.grad, model, data)
        param.grad = gradient

    optimizer.step()
    return


if __name__ == '__main__':

    args = params_init()

    # **VFL model save path**
    args.vfl_dir_save_model = f"./results/VFL_training_saved_models/VFL_{args.dataset}_{args.vfl_model}_saved_models"
    if not os.path.exists(args.vfl_dir_save_model):
        os.makedirs(args.vfl_dir_save_model)
    with open(f"{args.vfl_dir_save_model}/model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    VFL_train_main(args)


