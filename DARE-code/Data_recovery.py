"""
@author: tzh666
@context: Main function of data recovery based on MAE
"""
import os, copy, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from MAE_finetune import MAE_finetune_vit_base, mae_finetune_test
from models.DRModel import DRNet
from models.VFLModel import VFLBottomModel, model_init
from utils import get_random_img, set_loaders, bilinear_interpolation, psnr_compute, image_show, noise_mask
from params import params_init


# Measure the difference between two images restored by MAE
def MAE_loss(args, img0, mae_model, yy, label):
    """
    :param args: parameters
    :param img0: original image size=[batch_size,3,32,32]
    :param mae_model: MAE fine-tuned model
    :param yy: output of attack model size=[batch_size, n_class]
    :param label: ground-truth label
    :return: loss
    """
    ## For the true value `tru_`: only perform MAE once
    tru_pos = (0, label)
    # `_fea` represents the features output by the MAE model, used for computing the loss.
    _, _, tru_fea, tru_mask = mae_model(args, img0, origin_pos=tru_pos, remain_pos=tru_pos, mask_ratio=0.5)

    ## For the predicted value `pre_`: perform MAE sequentially by category.
    pre_fea = torch.zeros_like(tru_fea, device=args.device)
    data_dict = {}  # Store image data img0 in category order
    index_dict = {}  # Store corresponding indices in category order
    # Here, `each_class` represents the predicted class
    for each_class in range(args.n_class):
        data_dict[each_class] = []
        index_dict[each_class] = []
    # Add `img` and `index` to the specified lists in the dictionary
    for data_idx in range(img0.shape[0]):
        pre_label = int(max(0, min(yy[data_idx], args.n_class-1)))  # Ensure the range is between 0 and 4.
        data_dict[pre_label].append(img0[data_idx])
        index_dict[pre_label].append(data_idx)

    # Perform MAE per class
    class_length = 0
    total_loss = 0
    total_psnr = 0
    for each_class in range(args.n_class):
        if data_dict[each_class]:
            # list to tensor
            img = torch.zeros(len(data_dict[each_class]), data_dict[each_class][0].shape[0],
                              data_dict[each_class][0].shape[1], data_dict[each_class][0].shape[2], device=args.device)

            for i in range(len(data_dict[each_class])):
                img[i] = data_dict[each_class][i]
            loss, psnr, pre_fea_i, _ = mae_model(args, img, origin_pos=tru_pos, remain_pos=(0, each_class), mask_ratio=0.5)
            pre_fea[index_dict[each_class]] = pre_fea_i

            total_loss += loss
            total_psnr += psnr
            class_length += 1

    # compute loss
    # loss_func = nn.MSELoss()
    # pre_img = mae_model.unpatchify(pre_fea)
    # loss = loss_func(pre_fea, tru_fea)

    # compute psnr
    # psnr = psnr_compute(args, pre_img, img0)

    return total_loss/class_length, total_psnr/class_length


# Visualization of image restoration
def visualize(args, img, vfl_model, mae_model, attack_model):
    x = img.unsqueeze(0)

    # Compute the model's predicted values
    vfl_output = torch.transpose(vfl_model(x[:, :, :, 0:int(args.image_size/2)]).unsqueeze(0).unsqueeze(0), 0, 2)
    preds = attack_model(vfl_output)  # preds.shape=[batch_size, nclass+5]

    pre_label = (preds[:, -3]-preds[:, -1]/2)*args.patch_size
    pre_label = int(max(0, min(pre_label, args.n_class-1)))
    _, _, tru_fea, tru_mask = mae_model(args, x, origin_pos=(0, 0), remain_pos=(0, 0), mask_ratio=0.5)
    _, _, pre_fea, pre_mask = mae_model(args, x, origin_pos=(0, 0), remain_pos=(0, pre_label), mask_ratio=0.5)

    ## 1.show the original image
    plt.figure(figsize=(16, 16))
    plt.subplot(1, 5, 1)
    x = torch.einsum('nchw->nhwc', x)
    image_show(args, x[0], "original")

    ## 2.Display ground-truth mask and recovery image
    tru_mask = tru_mask.unsqueeze(-1).repeat([1, 1, mae_model.patch_embed.patch_size[0]**2*3])
    tru_mask = mae_model.unpatchify(tru_mask)
    tru_mask = torch.einsum('nchw->nhwc', tru_mask)
    im_masked = x*(1-tru_mask)
    plt.subplot(1, 5, 2)
    image_show(args, im_masked[0], "tru-masked")

    tru_fea = mae_model.unpatchify(tru_fea)
    tru_fea = torch.einsum('nchw->nhwc', tru_fea)
    im_paste = x*(1-tru_mask) + tru_fea*tru_mask
    plt.subplot(1, 5, 3)
    image_show(args, im_paste[0], "tru-recovery")

    ## 3. Display the mask and recovery image corresponding to the model's predicted output
    # To display, move the origin_pos part of the original image to remain_pos
    # Use deep copy to avoid changing the original values
    pre_pos = pre_label*args.patch_size  # Pixel coordinates corresponding to the image
    img_forshow = copy.deepcopy(x)
    img_forshow[:, pre_pos:int(pre_pos+args.image_size/2), :, :] = x[:, 0:int(args.image_size/2), :, :]

    # Plotting -> masked user image
    pre_mask = pre_mask.unsqueeze(-1).repeat([1, 1, mae_model.patch_embed.patch_size[0]**2*3])
    pre_mask = mae_model.unpatchify(pre_mask)
    pre_mask = torch.einsum('nchw->nhwc', pre_mask)
    im_masked = img_forshow*(1-pre_mask)
    plt.subplot(1, 5, 4)
    image_show(args, im_masked[0], "masked:0->"+str(pre_label))

    # Plotting -> attacker's recovered image
    pre_fea = mae_model.unpatchify(pre_fea)
    pre_fea = torch.einsum('nchw->nhwc', pre_fea)
    im_paste = img_forshow*(1-pre_mask) + pre_fea*pre_mask
    plt.subplot(1, 5, 5)
    image_show(args, im_paste[0], "result")

    plt.show()


def attack_test(args, vfl_model, mae_model, attack_model, testloader):
    for p in attack_model.parameters():
        p.requires_grad = False
    attack_model.eval()

    total_loss = 0
    total_psnr = 0

    with torch.no_grad():
        for b_id, b_data in enumerate(testloader):
            if b_id == 120:  # ~2000 in all
                break
            # img0.shape = (vfl_batch_size, 3, 32, 32)
            # img.shape = (vfl_batch_size, 3, 32, 16)
            # Testing using probable data of the client
            img0 = b_data[0].to(args.device)
            img0 = bilinear_interpolation(args, img0, size=224)
            img = img0[:, :, :, 0:int(args.image_size/2)]
            label = 0  # label->int The correct positions for this user are all (0, 0)

            vfl_output = torch.transpose(vfl_model(img).unsqueeze(0).unsqueeze(0), 0, 2)
            preds = attack_model(vfl_output)  # preds.shape=[batch_size, nclass+5]
            # ## Calculate classification loss
            # labels = torch.tensor([0] * 32, device=args.device)
            # loss1 = loss_function(preds, labels)
            # pred = torch.argmax(preds, dim=1)
            # test_acc += accuracy_score(pred, labels)
            # yy = pred * args.patch_size
            # print("test_pred:", pred)

            # ## Calculate coordinate loss -> L1
            # yy represents predicted top-left coordinates, projected into MAE coordinates. yy.shape: [batch_size]
            yy = (preds[:, -3] - preds[:, -1] / 2) * args.patch_size
            # # for printing purposes, you only need to compute `yy` because `xx` is 0.
            loss1 = torch.norm(yy)
            loss2, psnr = MAE_loss(args, img0, mae_model, yy, label)
            loss = loss1 + 100 * loss2
            total_loss += loss
            total_psnr += psnr

        print('===>[Data Recovery testing] loss:{}, psnr:{}'.format(total_loss/120, total_psnr/120))

        # Visualization: randomly select a test data point
        # img, _ = get_random_img(args, testloader)
        # img = bilinear_interpolation(img, size=224)
        # visualize(args, img, vfl_model, mae_model, attack_model)


def data_recovery_supervised_main(args):

    # ==> 1. Loading the attacker's VFL model
    attack_vfl_model = VFLBottomModel(args)
    attack_vfl_model.load_state_dict(torch.load(f'{args.vfl_dir_save_model}/{args.vfl_epochs}epochs_2users_bottom.pth'))
    attack_vfl_model.to(args.device)

    # ==> 2. Loading the MAE model
    mae_model = MAE_finetune_vit_base(img_size=args.image_size, patch_size=args.patch_size)
    mae_model.load_state_dict(torch.load(f'{args.mae_dir_save_model}/mae_finetuned_vitbase.pth'))
    mae_model.to(args.device)

    # ==> 3. Loading and processing data
    args.n_class = int(args.image_size / (2 * args.patch_size) + 1)
    attack_test_loader, _, attack_train_loader = set_loaders(args)

    # ==> 4. Training the attack model
    _, num_class = model_init(args)
    attack_model = DRNet(num_class=num_class).to(args.device)
    # optimizer = optim.Adam(attack_model.parameters(), lr=args.attack_lr, betas=(0.5, 0.999))
    optimizer = optim.SGD(attack_model.parameters(), lr=args.attack_lr)

    # start training，attack_model
    args.n_class = int(args.image_size/(2*args.patch_size)+1)
    for epoch in range(1+args.attack_epochs):
        for p in attack_model.parameters():
            p.requires_grad = True
        attack_model.train()
        total_loss = 0
        total_psnr = 0
        for b_id, b_data in enumerate(tqdm(attack_train_loader)):
            # Dividing by class. Each class represents a different partitioning method for the samples
            img0 = b_data[0].to(args.device)
            # Reshape the original data to 224x224
            img0 = bilinear_interpolation(args, img0, size=224)
            # Adding noise
            # Adding noise to the input data, used in computing VFL output and calculating losses.
            if args.attack_noise_type == "Noise":
                img0 = noise_mask(args, img0)
            for each_class in range(1):
                # img0.shape = (attack_batch_size, 3, 32, 32)
                # img.shape = (attack_batch_size, 3, 32, 16)
                # `img` is the input to the network.
                img = img0[:, :, :, each_class*args.patch_size:int(each_class*args.patch_size+args.image_size/2)]

                optimizer.zero_grad()

                # Calculate the output of the embedding layer
                # (attack_batch_size, 10)->(1, 1, attack_batch_size, 10)->(attack_batch_size, 1, 1, 10)
                vfl_output = torch.transpose(attack_vfl_model(img).unsqueeze(0).unsqueeze(0), 0, 2)

                preds = attack_model(vfl_output)  # preds.shape=[batch_size, nclass+5]

                # ## Calculate the classification loss for testing
                # labels = torch.tensor([each_class]*16, device=args.device)
                # loss1 = loss_function(preds, labels)
                # pred = torch.argmax(preds, dim=1)
                # train_acc += accuracy_score(pred, labels)

                # ## Calculate the coordinate regression loss -> L1
                # # (xx, yy) are the predicted top-left coordinates, projected into the MAE's coordinates.
                # #  yy.shape:[batch_size]
                xx = (preds[:, -4]-preds[:, -2]/2)*args.patch_size
                yy = (preds[:, -3]-preds[:, -1]/2)*args.patch_size
                loss1 = torch.norm(xx) + torch.norm(yy-each_class)
                ## Calculate the MAE loss corresponding to the image -> L2
                loss2, psnr = MAE_loss(args, img0, mae_model, yy, each_class)
                loss = loss1 + 100*loss2
                loss.backward()
                optimizer.step()

                total_loss += loss
                total_psnr += psnr

        print('===>[Recovery training] epoch:{}, training loss:{}, psnr:{}'.format(
            epoch, total_loss/(len(attack_train_loader)*args.n_class), total_psnr/(len(attack_train_loader)*args.n_class)))

        attack_test(args, attack_vfl_model, mae_model, attack_model, attack_test_loader)

        if args.save:
            torch.save(attack_model.state_dict(), f'{args.attack_dir_save_model}/loss1grad_{epoch}epoch.pth')


def data_recovery_unsupervised_main(args):

    # ==> 1. Loading the MAE model
    mae_model = MAE_finetune_vit_base(img_size=args.image_size, patch_size=args.patch_size)
    checkpoint = torch.load(f"./results/MAE_saved_models/MAE_official_pretrained_models/mae_official_pretrained_vitbase.pth",
                                map_location=args.device)
    # mae_model.load_state_dict(checkpoint['model'], strict=False)
    mae_model.load_state_dict(torch.load(f'{args.mae_dir_save_model}/mae_finetuned_vitbase.pth'))

    mae_model.to(args.device)

    # ==> 2. Load the attacker's VFL model
    attack_vfl_model = VFLBottomModel(args)
    attack_vfl_model.load_state_dict(
        torch.load(f'{args.vfl_dir_save_model}/{args.vfl_epochs}epochs_2users_bottom.pth'))
    attack_vfl_model.to(args.device)

    # ==> 3. Load and Process Auxiliary Data
    attack_test_loader, _, _ = set_loaders(args)

    # ==> 4. unsupervised recovery testing
    args.n_class = int(args.image_size/(2*args.patch_size)+1)

    total_loss = 0
    total_psnr = 0
    with torch.no_grad():
        for b_id, b_data in enumerate(tqdm(attack_test_loader)):
            if b_id == 62:  # ~2000 in all
                break
            img0 = b_data[0].to(args.device)
            img0 = bilinear_interpolation(args, img0, size=224)
            # Store the final recovery result
            pre_img = torch.zeros_like(img0, device=args.device)

            # Define a dictionary for storing category classifications to facilitate subsequent similarity calculations
            data_dict = {}  # Store image data img0 in order by category
            pred_dict = {}  # Store embedding layer information pred in order by category

            # each_class represents unknown label.
            # truth_class represents true label
            truth_class = 0
            for each_class in range(args.n_class):
                # true label is y=0, possible label is y=0 1...(n_class-1)
                _, _, pre_fea, _ = mae_model(args, img0, origin_pos=(0, truth_class), remain_pos=(0, each_class), mask_ratio=0.5)
                pre_fea = mae_model.unpatchify(pre_fea)
                data_dict[each_class] = pre_fea
                pred_dict[each_class] = attack_vfl_model(pre_fea)

            # Operate on each data item individually because `pred` may be different
            for data_id in range(len(img0)):
                cos_list = []
                for each_class1 in range(args.n_class):
                    cos = 0
                    for each_class2 in range(args.n_class):
                        if each_class2 != each_class1:
                            cos += torch.cosine_similarity(pred_dict[each_class1][data_id].view(-1),
                                                           pred_dict[each_class2][data_id].view(-1), dim=-1)
                    cos_list.append(cos)
                # print("data_id:{}, cos_list:{}".format(data_id, cos_list))

                # Find the index corresponding to the minimum value
                index = cos_list.index(min(cos_list))
                pre_img[data_id] = data_dict[index][data_id]

                # calculate psnr
                loss, psnr, _, _ = mae_model(args, img0[data_id].unsqueeze(0),
                                             origin_pos=(0, truth_class), remain_pos=(0, index), mask_ratio=0.5)
                total_loss += loss
                total_psnr += psnr

        print('[Unsupervised Recovery] loss:{}, psnr:{}'.format(total_loss/(62*args.vfl_batch_size),
                                                                total_psnr/(62*args.vfl_batch_size)))


if __name__ == '__main__':

    args = params_init()

    # VFL model save path
    args.vfl_dir_save_model = f"./results/VFL_training_saved_models/VFL_{args.dataset}_{args.vfl_model}_saved_models"

    # MAE finetuned model save path
    args.mae_dir_save_model = f"./results/MAE_saved_models/MAE_{args.dataset}_finetuned_models/finetuned_vitbase_id"

    # attack model save path
    if args.is_recovery_supervised:
        args.attack_dir_save_model = f"./results/Recovery_training_saved_models/Recovery_{args.dataset}_trained_models/vitbase_resnet20cnn5_id1"

        if not os.path.exists(args.attack_dir_save_model):
            os.makedirs(args.attack_dir_save_model)
        with open(f"{args.attack_dir_save_model}/attack_model_info.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Ensure consistent results across multiple code executions
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.is_recovery_supervised:
        data_recovery_supervised_main(args)
    else:
        data_recovery_unsupervised_main(args)

