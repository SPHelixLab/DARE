"""
@author: tzh666
@context: Extra functions. i.e. data processing.
"""
import random, math, os, sys, copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------------------------------------#
# ---------------------DATA PROCESSING-------------------------#
def set_loaders(args):
    """
    Data splitting, returning the dataloader for each part.
    Splitting method: VFL training data + test data + MAE fine-tuning/CNN training data.
    """
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if args.dataset == 'CIFAR10':
            train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
        else:
            train_dataset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform)

        # random shuffle
        index = list(range(len(train_dataset)))
        random.shuffle(index)
        vfl_index = index[:40000]
        aux_index = index[40000:]

    elif args.dataset == 'TinyImageNet':
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        data_dir = './data/TinyImageNet/'
        train_dataset = TinyImageNet(data_dir, train=True, transform=transform_train)
        test_dataset = TinyImageNet(data_dir, train=False, transform=transform_test)

        # random shuffle
        index = list(range(len(train_dataset)))
        random.shuffle(index)
        vfl_index = index[:70000]
        aux_index = index[70000:]

    elif args.dataset == 'Bank':
        data_dir = './data/bank/bank_cleaned.csv'
        total_dataset = Bank(data_dir)
        total_sample_num = len(total_dataset)
        train_set_num = int(total_sample_num * 0.8)
        test_set_num = total_sample_num - train_set_num

        train_dataset, test_dataset = random_split(total_dataset, [train_set_num, test_set_num])

        index = list(range(len(train_dataset)))
        random.shuffle(index)
        vfl_index = index[:20000]
        aux_index = index[20000:]

    else:
        raise Exception('The dataset is unknown!')

    # VFL training
    vfl_train_dataset = Subset(train_dataset, vfl_index)
    vfl_train_loader = DataLoader(dataset=vfl_train_dataset, batch_size=args.vfl_batch_size, shuffle=True)
    vfl_test_loader = DataLoader(dataset=test_dataset, batch_size=args.vfl_batch_size, shuffle=False)
    # MAE finetuning/CNN training
    cnn_train_dataset = Subset(train_dataset, aux_index)
    attack_aux_loader = DataLoader(dataset=cnn_train_dataset, batch_size=args.attack_batch_size, shuffle=True)

    return vfl_train_loader, vfl_test_loader, attack_aux_loader


# client data splitting
def split_data(args, data):
    """
    Data splitting, returning local data for each user. Horizontally partition the data, splitting it in half.
    """
    # image data
    if args.dataset in ['CIFAR10', 'CIFAR10', 'TinyImageNet']:
        if args.dataset == 'TinyImageNet':
            index = 112
        else:
            index = 16
        x_a = data[:, :, :, 0:index]
        x_b = data[:, :, :, index:index*2]

    # tabular data
    else:
        data = data.reshape(data.shape[0], 4, 5)[:, :, 0:4]
        data = data.unsqueeze(1).repeat(1, 3, 1, 1)
        index = 2
        x_a = data[:, :, :, 0:index]
        x_b = data[:, :, :, index:index*2]

    return x_a, x_b


# define size transformation, using bilinear interpolation
def bilinear_interpolation(args, img, size):
    interpolation = transforms.Resize(size=(size, size), interpolation=transforms.InterpolationMode.BILINEAR)
    if args.dataset == 'Bank':
        img = img.reshape(img.shape[0], 4, 5)[:, :, 0:4]
        img = interpolation(img).unsqueeze(1)
        img = img.repeat(1, 3, 1, 1)
    else:
        img = interpolation(img)
    return img


# -------------------------------------------------------------#
# -------------------------VISUALIZE---------------------------#
# randomly select an image from the dataset
def get_random_img(args, dataloader):
    img, label = next(iter(dataloader))
    return img[0].to(args.device), label


# image clipping
def clip_image(args, image, clip_min, clip_max):
    return torch.min(torch.max(torch.tensor(clip_min, device=args.device), image),
                     torch.tensor(clip_max, device=args.device))


# show the image
def image_show(args, img, title=''):
    # data preprocessing
    mean = torch.tensor([0.5, 0.5, 0.5], device=args.device)
    std = torch.tensor([0.5, 0.5, 0.5], device=args.device)
    img = (img * std + mean) * 255

    img = clip_image(args, img, 0.0, 255.0)
    img = img.cpu().detach().numpy().astype('int32')

    plt.imshow(img)
    plt.title(title, fontsize=16)
    plt.axis('off')


def tensor_to_image(args, img):
    # To convert a tensor with values [-1, 1] -> [0, 255]
    img = (img * 0.5 + 0.5) * 255
    img = clip_image(args, img, 0.0, 255.0)
    return img


def psnr_compute(args, img1, img2):
    total_len = len(img1)
    img1 = tensor_to_image(args, img1)
    img2 = tensor_to_image(args, img2)
    total_psnr = 0
    for i in range(total_len):
        mse = F.mse_loss(img1[i], img2[i])
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * math.log10(255 / math.sqrt(mse))

        total_psnr += psnr


    return total_psnr / total_len


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# TinyImageNet data preprocessing
# --------------------------------------------------------
class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if self.Train:
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


# --------------------------------------------------------
# Bank data preprocessing
# --------------------------------------------------------
class Bank(Dataset):
    def __init__(self, root):
        full_data_table = np.genfromtxt(root, delimiter=',')
        data = torch.from_numpy(full_data_table).float()

        self.samples = data[:, :-1]
        batch, columns = self.samples.size()
        permu_cols = torch.randperm(columns)
        self.samples = self.samples[:, permu_cols]
        min, _ = self.samples.min(dim=0)
        max, _ = self.samples.max(dim=0)
        self.feature_min = min
        self.feature_max = max
        self.samples = (self.samples - self.feature_min) / (self.feature_max - self.feature_min)

        self.labels = data[:, -1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]


# --------------------------------------------------------
# Defend methods
# --------------------------------------------------------
# Adding noise on the embedding layer
def noise_mask(args, input, noise_level=1):
    noise = torch.randn(input.size()) * noise_level
    noise = noise.to(args.device)
    output = input + noise
    return output

# Masking on gradients
def perturb_representation(input_gradient, model, ground_truth, pruning_rate=10):
    """
    Defense proposed in the Soteria paper.
    param:
        - input_gradient: the input_gradient
        - model: the ResNet-18 model
        - ground_truth: the benign image (for learning perturbed representation)
        - pruning_rate: the prune percentage
    Note: This implementation only works for ResNet-18
    """
    device = input_gradient.device
    gradient_ori = copy.deepcopy(input_gradient)

    gt_data = ground_truth.clone()
    gt_data.requires_grad = True

    # register forward hook to get intermediate layer output
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0]

        return hook

    # for bottom model
    handle = model.net.linear.register_forward_hook(get_activation('flatten'))
    out = model(gt_data)

    feature_graph = activation['flatten']

    deviation_target = torch.zeros_like(feature_graph)
    deviation_x_norm = torch.zeros_like(feature_graph)
    for f in range(deviation_x_norm.size(1)):
        deviation_target[:, f] = 1
        feature_graph.backward(deviation_target, retain_graph=True)
        deviation_f1_x = gt_data.grad.data
        deviation_x_norm[:, f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
        (feature_graph.data[:, f]) + 1e-10)
        model.zero_grad()
        gt_data.grad.data.zero_()
        deviation_target[:, f] = 0

    # prune r_i corresponding to smallest ||dr_i/dX||/||r_i||
    deviation_x_norm_sum = deviation_x_norm.sum(axis=0)
    thresh = np.percentile(deviation_x_norm_sum.flatten().cpu().numpy(), pruning_rate)
    mask = np.where(abs(deviation_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)

    # apply mask
    mask = torch.Tensor(mask).to(device)

    gradient_ori = torch.mul(gradient_ori, mask)

    handle.remove()

    return gradient_ori
