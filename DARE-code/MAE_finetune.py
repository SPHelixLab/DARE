"""
@author: tzh666
@context: MAE fine-tuning model structure
"""
import math, os, json, random, copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models.MAEModel import PatchEmbed, Block
from utils import set_loaders, bilinear_interpolation, psnr_compute, get_2d_sincos_pos_embed, get_random_img, image_show
from params import params_init


# The MAE model under VFL
class MAE_Finetune(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # -------------------------MAE encoder-------------------------#
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Storing encoder's positional and feature information
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        # transformer
        self.blocks = torch.nn.Sequential(*[Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # -------------------------MAE decoder-------------------------#
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        # Storing decoder's positional and feature information
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))

        self.decoder_pos_embed.stop_gradient = True  # fixed sin-cos embedding

        self.decoder_blocks = torch.nn.Sequential(
            *[Block(decoder_embed_dim, decoder_num_heads) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans)  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """(N, 3, H, W) -> (N, L, patch_size**2 *3)"""
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape([imgs.shape[0], 3, h, p, w, p])
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape([imgs.shape[0], h * w, p ** 2 * 3])
        return x

    def unpatchify(self, x):
        """(N, L, patch_size**2 *3) -> imgs: (N, 3, H, W)"""
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape([x.shape[0], h, w, p, p, 3])
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape([x.shape[0], 3, h * p, h * p])
        return imgs

# Essentially, this equates to using a different masking method here, corresponding to the VFL scenario.
# The network input remains the entire image because in this test scenario, no information from other parts is utilized.
# `remain` indicates whether the positions of the retained image after masking correspond to the original positions.
    def random_masking(self, args, x, origin_pos, remain_pos, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim = [1, 64, 192]
        len_keep = int(L * (1 - mask_ratio))

        # Mask positions are not randomly selected but determined based on existing positions.
        # Assume top-left coordinates (xx, yy) and divide using patches.
        # **users=2: xx=0, 0<=yy<n_class
        # client image size: (image_size, image_size/2)
        xx, yy = remain_pos[0], remain_pos[1]
        noise = torch.rand([N, L], device=args.device)  # noise in [0, 1]

        # Retain position -1 to ensure smaller numbers.
        for n in range(N):
            for i in range(L):
                if math.sqrt(L)*int(yy) <= i < math.sqrt(L)*(int(yy)+args.n_class-1):
                    noise[n, i] -= torch.tensor(1)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # Record positions in the original data using deepcopy.
        ori_xx, ori_yy = origin_pos[0], origin_pos[1]
        x[:, int(math.sqrt(L)*yy): int(math.sqrt(L)*(yy+args.n_class-1)), :] = \
            x[:, int(math.sqrt(L)*ori_yy): int(math.sqrt(L)*(ori_yy+args.n_class-1)), :]

        x_masked = x[torch.arange(N)[:, None], ids_keep]

        # `mask` serves as a flag indicating whether to retain/remove.
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=args.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = mask[torch.arange(N)[:, None], ids_restore]

        return x_masked, mask, ids_restore

    def forward_encoder(self, args, x, origin_pos, remain_pos, mask_ratio):

        # embed patches
        x = self.patch_embed(x)  # [1, 3, 32, 32] -> [1, 64, 192] --- 64=(32/4)*(32/4)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(args, x, origin_pos, remain_pos, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand([x.shape[0], -1, -1])
        x = torch.cat([cls_tokens, x], dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat([x.shape[0], ids_restore.shape[1]+1-x.shape[1], 1])
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token

        x_ = x_[torch.arange(x.shape[0])[:, None], ids_restore]  # unshuffle

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, args, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(axis=-1, keepdim=True)
            var = target.var(axis=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2  # MSE loss
        loss = loss.mean(axis=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        # compute the psnr of images
        pred = self.unpatchify(pred)
        mask = mask.unsqueeze(-1).repeat([1, 1, self.patch_embed.patch_size[0] ** 2 * 3])
        mask = self.unpatchify(mask)
        im_paste = imgs * (1 - mask) + pred * mask
        psnr = psnr_compute(args, imgs, im_paste)

        return loss, psnr

    def forward(self, args, imgs, origin_pos=(0, 0), remain_pos=(0, 0), mask_ratio=0.75):
        # encoder
        latent, mask, ids_restore = self.forward_encoder(args, imgs, origin_pos, remain_pos, mask_ratio)
        # decoder
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        # loss
        loss, psnr = self.forward_loss(args, imgs, pred, mask)

        return loss, psnr, pred, mask


# warmup cosine decay
class WarmupCosineLR(optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer,
                 warmup_start_lr,
                 end_lr,
                 warmup_epochs,
                 total_epochs,
                 last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.end_lr = end_lr

        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # linear warmup
        if self.last_epoch < self.warmup_epochs:
            lr = [(base_lr - self.warmup_start_lr) * float(self.last_epoch) / float(
                self.warmup_epochs) + self.warmup_start_lr for base_lr in self.base_lrs]
            return lr

        # cosine annealing decay
        progress = float(self.last_epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
        cosine_lr = max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))
        lr = [max(0.0, cosine_lr * (base_lr - self.end_lr) + self.end_lr) for base_lr in self.base_lrs]
        return lr


def MAE_finetune_vit_base(**kwargs):  # ViT-base
    model = MAE_Finetune(embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=512,
                         decoder_depth=8, decoder_num_heads=16, **kwargs)
    return model


# MAE image visualize--finetune
def visualize(args, img, model, mask_ratio):
    x = img.unsqueeze(0)

    _, _, pre, mask = model(args, x, mask_ratio=mask_ratio)
    pre = model.unpatchify(pre)
    pre = torch.einsum('nchw->nhwc', pre)

    mask = mask.unsqueeze(-1).repeat([1, 1, model.patch_embed.patch_size[0] ** 2 * 3])
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask)

    x = torch.einsum('nchw->nhwc', x)

    plt.figure(figsize=(12, 12))

    plt.subplot(1, 3, 1)
    image_show(args, x[0], "ft original")

    im_masked = x * (1 - mask)
    im_paste = x * (1 - mask) + pre * mask

    plt.subplot(1, 3, 2)
    image_show(args, im_masked[0], "masked")

    plt.subplot(1, 3, 3)
    image_show(args, im_paste[0], "reconstruction")

    plt.show()


# -------------------------------------------------------------#
# -------------------------FINE-TUNING-------------------------#
# finetune the main function
def mae_finetune_main(args):
    # ==> 1. Load pre-trained MAE model
    # Use the official pre-trained model, resize the input data to 224x224 because ViT only supports input data of size 224x224.
    ft_model = MAE_finetune_vit_base(img_size=args.image_size, patch_size=args.patch_size)
    checkpoint = torch.load(f"./results/MAE_saved_models/MAE_official_pretrained_models/mae_official_pretrained_vitbase.pth", map_location=args.device)
    ft_model.load_state_dict(checkpoint['model'], strict=False)
    ft_model.to(args.device)

    # ==> 2. Generate fine-tuning data for MAE
    args.n_class = int(args.image_size/(2*args.patch_size)+1)
    # Test using VFL's training data because the goal is to recover client data.
    cnn_test_loader, _, cnn_train_loader = set_loaders(args)

    mae_finetune_test(args, 0, ft_model, cnn_test_loader)

    # ==> 3. Setting optimizer and learning rate strategy
    opt = torch.optim.AdamW(params=ft_model.parameters(), lr=args.mae_lr, betas=(0.9, 0.95))
    lr_schedule = WarmupCosineLR(optimizer=opt,
                                 warmup_start_lr=args.mae_warm_start_lr,
                                 end_lr=args.mae_warm_end_lr,
                                 warmup_epochs=args.mae_warm_epochs,
                                 total_epochs=args.mae_finetune_epochs)

    # ==> 4. MAE fine-tuning
    # start training
    for epoch in range(1, args.mae_finetune_epochs+1):
        ft_model.train()
        total_loss = 0
        total_psnr = 0

        lr_now = opt.param_groups[0]['lr']
        print(f'===> [MAE fine-tune] epoch: {epoch}, lr: {lr_now:.6f}')

        # Fine-tune by batch
        for b_id, b_data in enumerate(tqdm(cnn_train_loader)):
            # imgs.shape = (batch_size, 3, 32, 32)
            imgs = b_data[0].to(args.device)
            # imgs.shape = (batch_size, 3, 224, 224)
            imgs = bilinear_interpolation(args, imgs, size=224)
            # Each class represents a different way of partitioning samples.
            for each_class in range(1):
                # During fine-tuning, coordinates are equal, all representing correct data.
                origin_position = remain_position = (0, each_class)

                loss, psnr, _, _ = ft_model(args, imgs, origin_pos=origin_position, remain_pos=remain_position, mask_ratio=0.5)
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss
                total_psnr += psnr

        lr_schedule.step()

        print(f"===> [MAE fine-tune training]: epoch:{epoch}, loss_avg: "
              f"{total_loss.item()/(len(cnn_train_loader)*args.n_class):.4f}, psnr_avg: {total_psnr/(len(cnn_train_loader)*args.n_class)}")

        mae_finetune_test(args, epoch, ft_model, cnn_test_loader)

        # save the model
        if args.save:
            torch.save(ft_model.state_dict(), f"{args.mae_dir_save_model}/mae_finetuned_vitbase.pth")


def mae_finetune_test(args, epoch, ft_model, cnn_test_loader):
    ft_model.eval()
    total_loss = 0
    total_psnr = 0

    with torch.no_grad():
        for b_id, b_data in enumerate(cnn_test_loader):
            imgs = b_data[0].to(args.device)
            imgs = bilinear_interpolation(args, imgs, size=224)
            loss, psnr, _, _ = ft_model(args, imgs, origin_pos=(0, 0), remain_pos=(0, 0), mask_ratio=0.5)
            total_loss += loss
            total_psnr += psnr
            if b_id == 200:
                break

        # Visualization: randomly select a test data point
        img, label = get_random_img(args, cnn_test_loader)
        img = bilinear_interpolation(img, size=224)
        visualize(args, img, ft_model, mask_ratio=0.5)

        print(f"===> [MAE fine-tune testing]: epoch:{epoch}, loss: {total_loss.item()/200}, psnr: {total_psnr/200}")


if __name__ == '__main__':

    args = params_init()

    # MAE model save path
    # id used to differentiate models, detailed parameters in model_info.txt
    args.mae_dir_save_model = f"./results/MAE_saved_models/MAE_{args.dataset}_finetuned_models/finetuned_vitbase_id"

    if not os.path.exists(args.mae_dir_save_model):
        os.makedirs(args.mae_dir_save_model)
    with open(f"{args.mae_dir_save_model}/finetuned_model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Ensure consistent results across multiple code executions
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mae_finetune_main(args)


