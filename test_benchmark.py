import argparse
import os
import torch
import numpy as np
from utils.loss import BceDiceLoss
from sklearn.metrics import confusion_matrix
from utils.filter import normalize_image, low_pass_filter
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from model.LightMed import LightMed
def parse_args():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--model_paths', nargs='+', required=True, help='Paths to model files')
    parser.add_argument('--test_dataset_paths', nargs='+', required=True, help='Paths to test datasets')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--in_channels', type=int, default=3, help='Input channels')
    parser.add_argument('--out_channels', type=int, default=1, help='Output channels')
    parser.add_argument('--r', type=int, default=64, help='Radius for low pass filter')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for predictions')

    return parser.parse_args()

class CustomImageMaskDataset(Dataset):
    def __init__(self, images, masks, image_transform=None):
        self.images = images
        self.masks = masks
        self.image_transform = image_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        image = image.transpose(1, 2, 0)
        mask = mask.transpose(1, 2, 0)
        if self.image_transform:
            aug = self.image_transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        mask = np.where(mask > 0, 1., 0.)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        return image, mask

def process_dataset(test_dataset, r):
    x_test = []
    y_test = []
    for images, masks in test_dataset:
        images = normalize_image(np.array(images))
        x_test.append(np.stack(images))
        y_test.append(np.stack(masks))
    return np.array(x_test), np.array(y_test)

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

def test_one_epoch(test_loader, model, criterion, threshold, device):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for img, msk in test_loader:
            img, msk = img.to(device, dtype=torch.float32), msk.to(device, dtype=torch.float32)
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().numpy()
            gts.append(msk)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze(1).cpu().numpy()
            preds.append(out)
        preds = np.concatenate(preds, axis=0).reshape(-1)
        gts = np.concatenate(gts, axis=0).reshape(-1)
        y_pre = np.where(preds >= threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)
        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion.ravel()
        accuracy = (TN + TP) / np.sum(confusion) if np.sum(confusion) != 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        f1_or_dsc = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
        f2_score = (5 * TP) / (5 * TP + 4 * FN + FP) if (5 * TP + 4 * FN + FP) != 0 else 0
        miou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
        log_info = f'loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, f2_score: {f2_score}, ' \
                   f'accuracy: {accuracy}, specificity: {specificity}, sensitivity: {sensitivity}'
        print(log_info)
    return np.mean(loss_list)

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = BceDiceLoss()

    
    test_transforms = A.Compose([
        A.Resize(width=args.image_size, height=args.image_size, p=1.0)
    ])

    # load data
    test_datasets = []
    for dataset_path in args.test_dataset_paths:
        X_test = np.load(os.path.join(dataset_path, 'images_test.npy'))
        y_test = np.load(os.path.join(dataset_path, 'masks_test.npy'))
        test_dataset = CustomImageMaskDataset(X_test, y_test, test_transforms)
        test_datasets.append(test_dataset)

    # processing data
    processed_datasets = []
    for test_dataset in test_datasets:
        x_test, y_test = process_dataset(test_dataset, args.r)
        processed_datasets.append((x_test, y_test))

    # Creat DataLoader
    test_dataloaders = []
    for x_test, y_test in processed_datasets:
        dataset = CustomDataset(x_test, y_test)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        test_dataloaders.append(dataloader)

    for path in args.model_paths:
        model = LightMed(args.in_channels, args.out_channels).to(device)
        model.load_state_dict(torch.load(path))
        model = model.to(device)
        print(f"Testing with model from: {path}")
        for scenario_idx, dataloader in enumerate(test_dataloaders):
            print(f"Scenario {scenario_idx}:")
            test_one_epoch(dataloader, model, criterion, args.threshold, device)

if __name__ == '__main__':

    main()
