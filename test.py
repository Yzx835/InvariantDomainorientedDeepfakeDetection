import os
import sys
# sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader import MyDataset_FF
from model_Xception import xception
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='xception')
parser.add_argument("--datapath", type=str, default='./data')
parser.add_argument("--splitpath", type=str, default='./split')
parser.add_argument("--modelpath", type=str)
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--acc_txt', type=str, default='acc_new.txt', help='txtlog')
parser.add_argument("--batchsize", type=int, default=48)
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


def test():
    device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
    model = xception(num_classes=2)
    model.load_state_dict(torch.load(args.modelpath
        ,map_location=lambda storage, loc:storage.cuda()
        ), strict=False)
    model.to(device)

    dataset = MyDataset_FF(data_dir = args.datapath, split_path = args.splitpath,
                        data_type=['raw', 'deepfakes', 'faceswap', 'face2face','neuraltextures'],
                        train_val_test='test') 
    
    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=16)

    cirterion = nn.CrossEntropyLoss()

    ys = []
    scores = []

    with torch.no_grad():

        # domain adapation
        model.train()
        
        total_loss, correct, total = 0, 0, 0
        tqdm_bar = tqdm(loader)

        for i, data in enumerate(tqdm_bar):

            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            
            loss = cirterion(outputs, labels)

            prob = F.softmax(outputs, dim=1)
            
            for _i in range(len(labels)):
                ys.append(int(labels[_i]))
                scores.append(float(prob[_i][1]))

            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum()
            total_loss += loss.sum()

            tqdm_bar.set_description("loss: %.6f, correct: %.4f" % (total_loss / total, correct.float() / total))

        total_loss /= total
        acc = float(correct.float() / total)

        ys = np.array(ys)
        scores = np.array(scores)
        fpr, tpr, thresholds = metrics.roc_curve(ys, scores)
        auc = metrics.auc(fpr, tpr)
        print(auc, acc)
        with open(args.acc_txt, 'a') as f:
            f.write("%s,%f,%f\n" % (args.modelpath, acc, auc))

        # plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('roc.png')


if __name__ == '__main__':
    test()
