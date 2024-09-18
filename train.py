import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import albumentations as A
from utils.filter import normalize_image, low_pass_filter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils.loss import BceDiceLoss
import os
from model.LightMed import LightMed
import math
from datetime import datetime

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')

    # Global parameters
    parser.add_argument('--r', type=int, default=64, help='Radius parameter for low-pass filter')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the input images')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=1, help='Number of output channels')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')

    # Additional parameters
    parser.add_argument('--val_interval', type=int, default=40, help='Validation interval')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification')
    parser.add_argument('--work_dir', type=str, default='./checkpoint/', help='Working directory')

    # Optimizer parameters
    parser.add_argument('--opt', type=str, default='AdamW', help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay (L2 penalty)')

    # For Adam and its variants
    parser.add_argument('--betas', type=float, nargs=2, default=None, help='Betas for Adam optimizer')
    parser.add_argument('--eps', type=float, default=None, help='Term added to denominator to improve numerical stability')
    parser.add_argument('--amsgrad', action='store_true', help='AMSGrad variant for Adam')

    # For SGD
    parser.add_argument('--momentum', type=float, default=None, help='Momentum factor')
    parser.add_argument('--dampening', type=float, default=None, help='Dampening for momentum')
    parser.add_argument('--nesterov', action='store_true', help='Enables Nesterov momentum')

    # For Adadelta
    parser.add_argument('--rho', type=float, default=None, help='Coefficient used for computing a running average of squared gradients')

    # For Adagrad
    parser.add_argument('--lr_decay', type=float, default=None, help='Learning rate decay')

    # For RMSprop
    parser.add_argument('--alpha', type=float, default=None, help='Smoothing constant')
    parser.add_argument('--centered', action='store_true', help='Compute the centered RMSProp')

    # For Rprop
    parser.add_argument('--etas', type=float, nargs=2, default=None, help='Pair of (etaminus, etaplus)')
    parser.add_argument('--step_sizes', type=float, nargs=2, default=None, help='Minimal and maximal allowed step sizes')

    # For ASGD
    parser.add_argument('--lambd', type=float, default=None, help='Decay term')
    parser.add_argument('--t0', type=float, default=None, help='Point at which to start averaging')

    # Scheduler parameters
    parser.add_argument('--sch', type=str, default='CosineAnnealingLR', help='Scheduler to use')
    parser.add_argument('--gamma', type=float, default=None, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--last_epoch', type=int, default=-1, help='The index of last epoch')

    # For StepLR
    parser.add_argument('--step_size', type=int, default=None, help='Period of learning rate decay')

    # For MultiStepLR
    parser.add_argument('--milestones', type=int, nargs='+', default=None, help='List of epoch indices')

    # For CosineAnnealingLR
    parser.add_argument('--T_max', type=int, default=None, help='Maximum number of iterations')
    parser.add_argument('--eta_min', type=float, default=None, help='Minimum learning rate')

    # For ReduceLROnPlateau
    parser.add_argument('--mode', type=str, default=None, help='One of min, max')
    parser.add_argument('--factor', type=float, default=None, help='Factor by which the learning rate will be reduced')
    parser.add_argument('--patience', type=int, default=None, help='Number of epochs with no improvement after which learning rate will be reduced')
    # parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for measuring the new optimum')
    parser.add_argument('--threshold_mode', type=str, default=None, help='One of rel, abs')
    parser.add_argument('--cooldown', type=int, default=None, help='Number of epochs to wait before resuming normal operation after lr has been reduced')
    parser.add_argument('--min_lr', type=float, default=None, help='A lower bound on the learning rate')
    parser.add_argument('--eps_scheduler', type=float, default=None, help='Minimal decay applied to lr')

    # For CosineAnnealingWarmRestarts
    parser.add_argument('--T_0', type=int, default=None, help='Number of iterations for the first restart')
    parser.add_argument('--T_mult', type=int, default=None, help='A factor increases T_i after a restart')

    # For WP_MultiStepLR and WP_CosineLR
    parser.add_argument('--warm_up_epochs', type=int, default=None, help='Number of warm-up epochs')

    args = parser.parse_args()
    return args

# Custom Dataset for DataLoader
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = torch.from_numpy(self.x_data[index])
        y = torch.from_numpy(self.y_data[index])
        return x, y

# DataLoader creation function
def create_dataloader(x_data, y_data, batch_size, shuffle):
    dataset = CustomDataset(x_data, y_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Processing dataset function
def process_dataset(dataset, r):
    x_processed, y_processed = [], []
    for images, masks in dataset:
        images = normalize_image(np.array(images))
        x_processed.append(np.stack(images))
        y_processed.append(np.stack(masks))
    return np.array(x_processed), np.array(y_processed)

# Setting configuration class
class SettingConfig:
    def __init__(self, args):
        self.criterion = BceDiceLoss()
        self.num_classes = args.out_channels
        self.input_size_h = args.image_size
        self.input_size_w = args.image_size
        self.input_channels = args.in_channels
        self.batch_size = args.batch_size
        self.epochs = args.num_epochs
        self.val_interval = args.val_interval
        self.threshold = args.threshold
        self.seed = args.seed
        self.work_dir = args.work_dir
        self.opt = args.opt
        self.lr = args.lr
        self.r = args.r
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Optimizer parameters
        assert self.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

        # Set optimizer-specific parameters
        if self.opt == 'Adadelta':
            self.rho = args.rho if args.rho is not None else 0.9
            self.eps = args.eps if args.eps is not None else 1e-6
            self.weight_decay = args.weight_decay if args.weight_decay is not None else 0.05
        elif self.opt == 'Adagrad':
            self.lr_decay = args.lr_decay if args.lr_decay is not None else 0
            self.eps = args.eps if args.eps is not None else 1e-10
            self.weight_decay = args.weight_decay if args.weight_decay is not None else 0.05
        elif self.opt == 'Adam':
            self.betas = tuple(args.betas) if args.betas is not None else (0.9, 0.999)
            self.eps = args.eps if args.eps is not None else 1e-8
            self.weight_decay = args.weight_decay if args.weight_decay is not None else 0.0001
            self.amsgrad = args.amsgrad
        elif self.opt == 'AdamW':
            self.betas = tuple(args.betas) if args.betas is not None else (0.9, 0.999)
            self.eps = args.eps if args.eps is not None else 1e-8
            self.weight_decay = args.weight_decay if args.weight_decay is not None else 1e-2
            self.amsgrad = args.amsgrad
        elif self.opt == 'Adamax':
            self.betas = tuple(args.betas) if args.betas is not None else (0.9, 0.999)
            self.eps = args.eps if args.eps is not None else 1e-8
            self.weight_decay = args.weight_decay if args.weight_decay is not None else 0
        elif self.opt == 'ASGD':
            self.lambd = args.lambd if args.lambd is not None else 1e-4
            self.alpha = args.alpha if args.alpha is not None else 0.75
            self.t0 = args.t0 if args.t0 is not None else 1e6
            self.weight_decay = args.weight_decay if args.weight_decay is not None else 0
        elif self.opt == 'RMSprop':
            self.momentum = args.momentum if args.momentum is not None else 0
            self.alpha = args.alpha if args.alpha is not None else 0.99
            self.eps = args.eps if args.eps is not None else 1e-8
            self.centered = args.centered
            self.weight_decay = args.weight_decay if args.weight_decay is not None else 0
        elif self.opt == 'Rprop':
            self.etas = tuple(args.etas) if args.etas is not None else (0.5, 1.2)
            self.step_sizes = tuple(args.step_sizes) if args.step_sizes is not None else (1e-6, 50)
        elif self.opt == 'SGD':
            self.momentum = args.momentum if args.momentum is not None else 0.9
            self.weight_decay = args.weight_decay if args.weight_decay is not None else 0.05
            self.dampening = args.dampening if args.dampening is not None else 0
            self.nesterov = args.nesterov
        else:
            # default optimizer is SGD
            self.momentum = 0.9
            self.weight_decay = 0.05
            self.dampening = 0
            self.nesterov = False

        # Scheduler parameters
        self.sch = args.sch
        assert self.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                            'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'

        # Set scheduler-specific parameters
        self.gamma = args.gamma if args.gamma is not None else 0.1  # Common default gamma
        self.last_epoch = args.last_epoch

        if self.sch == 'StepLR':
            self.step_size = args.step_size if args.step_size is not None else self.epochs // 5
        elif self.sch == 'MultiStepLR':
            self.milestones = args.milestones if args.milestones is not None else [60, 120, 150]
        elif self.sch == 'ExponentialLR':
            pass  # gamma is already set
        elif self.sch == 'CosineAnnealingLR':
            self.T_max = args.T_max if args.T_max is not None else 50
            self.eta_min = args.eta_min if args.eta_min is not None else 0.00001
        elif self.sch == 'ReduceLROnPlateau':
            self.mode = args.mode if args.mode is not None else 'min'
            self.factor = args.factor if args.factor is not None else 0.1
            self.patience = args.patience if args.patience is not None else 10
            self.threshold = args.threshold if args.threshold is not None else 0.0001
            self.threshold_mode = args.threshold_mode if args.threshold_mode is not None else 'rel'
            self.cooldown = args.cooldown if args.cooldown is not None else 0
            self.min_lr = args.min_lr if args.min_lr is not None else 0
            self.eps_scheduler = args.eps_scheduler if args.eps_scheduler is not None else 1e-8
        elif self.sch == 'CosineAnnealingWarmRestarts':
            self.T_0 = args.T_0 if args.T_0 is not None else 50
            self.T_mult = args.T_mult if args.T_mult is not None else 2
            self.eta_min = args.eta_min if args.eta_min is not None else 1e-6
        elif self.sch == 'WP_MultiStepLR':
            self.warm_up_epochs = args.warm_up_epochs if args.warm_up_epochs is not None else 10
            self.milestones = args.milestones if args.milestones is not None else [125, 225]
        elif self.sch == 'WP_CosineLR':
            self.warm_up_epochs = args.warm_up_epochs if args.warm_up_epochs is not None else 20
        else:
            pass

# Functions for training, validation, and testing
def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, device):
    model.train()
    loss_list = []
    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.to(device), targets.to(device)
        out = model(images).to(device)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    print(f'Train: epoch {epoch}, loss: {np.mean(loss_list):.4f}')
    scheduler.step()

def val_one_epoch(test_loader, model, criterion, epoch, config):
    device = config.device
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in (test_loader):
            img, msk = data
            img, msk = img.to(device), msk.to(device)
            out = model(img).to(device)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        f2_score = float(5 * TP) / float(5 * TP + 4 * FN + FP) if float(5 * TP + 4 * FN + FP) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, f2_score: {f2_score}, accuracy: {accuracy}, specificity: {specificity}, sensitivity: {sensitivity}'

        print(log_info)
    else:
        log_info = f'Val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)

    return np.mean(loss_list)

def test_one_epoch(test_loader, model, criterion, config):
    device = config.device
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate((test_loader)):
            img, msk = data
            img, msk = img.to(device), msk.to(device)
            out = model(img).to(device)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        f2_score = float(5 * TP) / float(5 * TP + 4 * FN + FP) if float(5 * TP + 4 * FN + FP) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, f2_score: {f2_score}, accuracy: {accuracy}, specificity: {specificity}, sensitivity: {sensitivity}'

        print(log_info)

    return np.mean(loss_list)

def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=config.lr,
            lr_decay=config.lr_decay,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr=config.lr,
            lambd=config.lambd,
            alpha=config.alpha,
            t0=config.t0,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.eps,
            centered=config.centered,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr=config.lr,
            etas=config.etas,
            step_sizes=config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dampening=config.dampening,
            nesterov=config.nesterov
        )
    else:  # default optimizer is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.05,
        )

def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                          'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps_scheduler
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len(
            [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler

def main(config):
    device = config.device
    print('#----------Preparing Models----------#')
    model = LightMed(config.input_channels, config.num_classes).to(device)
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    resume_model = os.path.join(config.work_dir, 'latest.pth')

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cuda'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'Resuming model from {resume_model}. Resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        print(log_info)

    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Augmentations
    train_transforms = A.Compose([
        A.Resize(width=config.input_size_w, height=config.input_size_h, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.3, rotate_limit=120, p=0.5)
    ])

    val_test_transforms = A.Compose([
        A.Resize(width=config.input_size_w, height=config.input_size_h, p=1.0)
    ])

    class CustomImageMaskDataset(Dataset):
        def __init__(self, X, y, transform=None):
            self.images = X
            self.masks = y
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx].transpose(1, 2, 0)  # H,W,C
            mask = self.masks[idx].transpose(1, 2, 0)    # H,W,C

            if self.transform:
                aug = self.transform(image=image, mask=mask)
                image, mask = aug['image'], aug['mask']

            mask = np.where(mask > 0, 1.0, 0.0)
            return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)  # C,H,W

    # Load datasets
    x_train = np.load('./ISIC_2018/images_train.npy')
    y_train = np.load('./ISIC_2018/masks_train.npy')

    # Split train/validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1/8, random_state=config.seed)

    # Create datasets
    train_dataset = CustomImageMaskDataset(x_train, y_train, transform=train_transforms)
    val_dataset = CustomImageMaskDataset(x_val, y_val, transform=val_test_transforms)

    # Process train/val datasets
    x_train_processed, y_train_processed = process_dataset(train_dataset, config.r)
    x_val_processed, y_val_processed = process_dataset(val_dataset, config.r)


    # Create DataLoaders
    train_dataloader = create_dataloader(x_train_processed, y_train_processed, config.batch_size, shuffle=True)
    val_dataloader = create_dataloader(x_val_processed, y_val_processed, config.batch_size, shuffle=False)

    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()

        train_one_epoch(
            train_dataloader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            device
        )

        loss = val_one_epoch(
            val_dataloader,
            model,
            criterion,
            epoch,
            config
        )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(config.work_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(config.work_dir, 'latest.pth'))
    print("-----------SUMMARY-------------")
    model.load_state_dict(torch.load(os.path.join(config.work_dir, 'best.pth')))

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

if __name__ == '__main__':
    args = parse_args()
    config = SettingConfig(args)
    main(config)



