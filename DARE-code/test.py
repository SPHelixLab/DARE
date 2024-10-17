import torch, copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from params import params_init
from utils import set_loaders, bilinear_interpolation, image_show
from MAE_finetune import MAE_finetune_vit_base


# Visualize the test results
def visualize_test(args):
    mae_model = MAE_finetune_vit_base(img_size=args.image_size, patch_size=args.patch_size)
    mae_model.load_state_dict(torch.load(f'{args.mae_dir_save_ft_model}/mae_finetuned_vitbase.pth'))
    mae_model.to(args.device)

    # ==> 2. Loading and processing data
    _, test_loader, _ = set_loaders(args)

    # ==> 3. visualize
    args.n_class = int(args.image_size/(2*args.patch_size)+1)
    image_dir = f"./results/Extra_experiments_results/{args.dataset}_show_images"
    pre_label = 1
    with torch.no_grad():
        batch = 0
        for _, b_img in enumerate(tqdm(test_loader)):
            if batch < 15:
                batch += 1
                continue
            imgs = b_img[0].to(args.device)
            imgs = bilinear_interpolation(args, imgs, size=224)
            _, _, tru_feas, tru_masks = mae_model(args, imgs, origin_pos=(0, 0), remain_pos=(0, 0), mask_ratio=0.5)
            _, _, pre_feas, pre_masks = mae_model(args, imgs, origin_pos=(0, 0), remain_pos=(0, pre_label), mask_ratio=0.5)
            for i in range(len(b_img[0])):
                x = imgs[i].unsqueeze(0)
                tru_fea = tru_feas[i].unsqueeze(0)
                tru_mask = tru_masks[i].unsqueeze(0)
                pre_fea = pre_feas[i].unsqueeze(0)
                pre_mask = pre_masks[i].unsqueeze(0)

                ## 1.show the original image
                number = batch * args.vfl_batch_size + i
                plt.figure(figsize=(16, 16))
                x = torch.einsum('nchw->nhwc', x)
                image_show(args, x[0], "original")
                plt.savefig(f'{image_dir}/image{number}_original.png', dpi=500)

                ## 2.show the supervised recovery result
                tru_mask = tru_mask.unsqueeze(-1).repeat([1, 1, mae_model.patch_embed.patch_size[0] ** 2 * 3])
                tru_mask = mae_model.unpatchify(tru_mask)
                tru_mask = torch.einsum('nchw->nhwc', tru_mask)
                tru_fea = mae_model.unpatchify(tru_fea)
                tru_fea = torch.einsum('nchw->nhwc', tru_fea)
                im_paste = x * (1 - tru_mask) + tru_fea * tru_mask
                image_show(args, im_paste[0], "sup-recovery")
                plt.savefig(f'{image_dir}/image{number}_sup_recovery.png', dpi=500)

                ## 3.show the unsupervised recovery result
                pre_pos = pre_label * args.patch_size  # Pixel coordinates corresponding to the image
                img_forshow = copy.deepcopy(x)
                img_forshow[:, pre_pos:int(pre_pos + args.image_size / 2), :, :] = x[:, 0:int(args.image_size / 2), :, :]

                # Plotting -> masked client image
                pre_mask = pre_mask.unsqueeze(-1).repeat([1, 1, mae_model.patch_embed.patch_size[0] ** 2 * 3])
                pre_mask = mae_model.unpatchify(pre_mask)
                pre_mask = torch.einsum('nchw->nhwc', pre_mask)
                pre_fea = mae_model.unpatchify(pre_fea)
                pre_fea = torch.einsum('nchw->nhwc', pre_fea)
                im_paste = img_forshow * (1 - pre_mask) + pre_fea * pre_mask
                image_show(args, im_paste[0], "unsup-recovery")
                plt.savefig(f'{image_dir}/image{number}_unsup_recovery.png', dpi=500)

            batch += 1

# Used for debugging
if __name__ == '__main__':
    args = params_init()

    # MAE finetuned model save path
    args.mae_dir_save_ft_model = f"./results/MAE_saved_models/MAE_{args.dataset}_finetuned_models/finetuned_vitbase_id"

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    visualize_test(args)