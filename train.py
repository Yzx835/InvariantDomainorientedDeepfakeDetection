import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader
from dataloader import MyDataset_FF, MyDataloader
from model_Xception import xception
import random
from my_transforms import create_train_transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='xception')
parser.add_argument("--datapath", type=str, default='./data')
parser.add_argument("--splitpath", type=str, default='./split')
parser.add_argument("--region_erase_path1", type=str, default='./region_erase_5.json')
parser.add_argument("--region_erase_path2", type=str, default='./region_erase_68.json')
parser.add_argument("--epoch", type=int, default=15)
parser.add_argument("--batchsize", type=int, default=48)
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--DABN', type=int, default=3,help='DABN')

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

def make_log_dir():
    logs = {
        'T': args.model,
        'DABN': args.DABN, 
        'batchsize' : args.batchsize,
        'epoch': args.epoch,
    }
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + '-' + str(logs[key]) + '+'
    dir_name = 'logs/' + dir_name
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    return dir_name

log_dir = make_log_dir()

cirterion = nn.CrossEntropyLoss()

def save_image(image_tensor, save_file):
    image_tensor.detach_()
    image_tensor = image_tensor[:16]
    image_tensor[:, 0, :, :] = image_tensor[:, 0, :, :] * 0.229 + 0.485
    image_tensor[:, 1, :, :] = image_tensor[:, 1, :, :] * 0.224 + 0.456
    image_tensor[:, 2, :, :] = image_tensor[:, 2, :, :] * 0.225 + 0.406
    torchvision.utils.save_image(image_tensor, save_file, nrow=4)


def train():
    model = xception(num_classes=2)
    model = model.cuda()
    dataset_raw = MyDataset_FF(data_dir = args.datapath, split_path = args.splitpath,
                data_type = ['raw', 'deepfakes', 'faceswap', 'face2face','neuraltextures'], 
                train_val_test='train', erase_rate = 0, transforms = create_train_transforms(224))
    dataset_raw.reset()
    
    dataset_swap = MyDataset_FF(data_dir = args.datapath, split_path = args.splitpath,
                data_type =['raw', 'deepfakes_swap', 'faceswap_swap', 'face2face','neuraltextures'], 
                train_val_test ='train', erase_rate = 0, transforms = create_train_transforms(224))
    dataset_swap.reset()

    dataset_LswapL = MyDataset_FF(data_dir = args.datapath, split_path = args.splitpath,
                region_erase_path1 = args.region_erase_path1 , region_erase_path2 = args.region_erase_path2,
                data_type = ['raw', 'deepfakes', 'faceswap', 'face2face','neuraltextures', 'deepfakes_swap', 'faceswap_swap'], 
                train_val_test='train', erase_rate = 1, transforms = create_train_transforms(224))
    dataset_LswapL.reset()
    
    dataset_list = [dataset_raw, dataset_swap, dataset_LswapL]
    loaders = []
    for i in range(len(dataset_list)):
        dataset_ = dataset_list[i]
        loader = DataLoader(dataset_, batch_size=args.batchsize//args.DABN, shuffle=True, num_workers=4, drop_last=True)
        loaders.append(loader)
        
    train_loader = MyDataloader(loaders)

    
    val_dataset = MyDataset_FF(data_dir = args.datapath, split_path = args.splitpath,
                data_type = ['raw', 'deepfakes', 'faceswap', 'face2face','neuraltextures'], 
                train_val_test = 'val')
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize//args.DABN, shuffle=True, num_workers=4, drop_last=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 8, 15, 25, 40, 60, 90, 140], gamma=1/3)
    
    model_best = 0
    best_epoch = -1
    for epoch in range(1,args.epoch+1):
        print('EPOCH: ', epoch)
        print(optimizer)
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        tqdm_bar = tqdm(range(len(train_loader)))
        for i in tqdm_bar:
            data = train_loader.get_next()
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda() 
            
            outputs = model(inputs)
            
            loss = cirterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum()
            total_loss += loss.sum()
            if i % 100 == 0:
                save_image(inputs, 'inputs.png')
            tqdm_bar.set_description("loss: %.6f, correct: %.4f" % (total_loss / total, correct.float() / total))
        total_loss /= total
        correct = correct.float() / total
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, '%s_%d.pth' % (args.model, epoch)))
        with open(os.path.join(log_dir, 'train_log.txt'), 'a') as f:
            f.write('epoch: %d, loss: %.6f, acc: %.4f %%\n' % (epoch, total_loss, correct))
            
            
        print("Val")
        with torch.no_grad():
            model.eval()
            total_loss, correct, total = 0, 0, 0
            tqdm_bar = tqdm(val_loader)
            for i, data in enumerate(tqdm_bar):
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                outputs = model(inputs)
                
                loss = cirterion(outputs, labels)
                predicted = torch.max(outputs, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum()
                total_loss += loss.sum()
                tqdm_bar.set_description("loss: %.6f, correct: %.4f" % (total_loss / total, correct.float() / total))
        total_loss /= total
        correct = correct.float() / total
        
        if correct > model_best:
            model_best = correct
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(log_dir, '%s_best.pth' % args.model))
        with open(os.path.join(log_dir, 'test_log.txt'), 'a') as f:
            f.write('epoch: %d, loss: %.6f, acc: %.4f %%, model_best: %d\n' % (epoch, total_loss, correct, best_epoch))
        
        scheduler.step()

if __name__ == '__main__':
    train()