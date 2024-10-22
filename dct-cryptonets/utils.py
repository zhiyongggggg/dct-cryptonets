""" Utility classes and functions """

import random
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix


class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type='softmax'):
        super(BaselineTrain, self).__init__()
        try:
            self.feature = model_func()
        except TypeError:
            self.feature = model_func

        self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
        self.classifier.bias.data.fill_(0)
        self.loss_type = loss_type
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_prec1_val = None

    def forward(self, x):
        x = Variable(x.cuda())
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return out, scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        y = Variable(y.cuda())
        return self.loss_fn(scores, y)

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0

        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.data[0]

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))

    def test_loop(self, val_loader):
        return -1  # no validation, just save model during iteration


class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopper:
    """ Early stopping based on validation loss tracking """
    def __init__(self, patience=1, threshold=0.0):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.threshold):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(params, optimizer, epoch):
    epoch += 1
    if epoch in params.schedule:
        params.lr *= params.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = params.lr
    return params


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def plot_examples(params, train_data):
    rand_idx = random.sample(range(len(train_data)), k=16)
    plt.figure(figsize=(10, 10))

    for i, idx in enumerate(rand_idx):
        img, label = train_data[idx]
        # the image tensor's range is not between 0 and 1,
        # so we have to temporarily scale the tensor values into range 0 and 1 to prevent error.
        img = (img - img.min()) / (img.max() - img.min())
        img_class = train_data.classes[label]

        plt.subplot(4, 4, i + 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Class : {img_class}", fontsize=10)
    plt.savefig(f'{params.checkpoint_dir}/example_images.png', dpi=400)
    return


def pred_classes(params, model, test_data):
    predicted_labels = []
    actual_labels = []

    model.eval()
    with torch.no_grad():  # We are using no_grad instead of inference_mode for better compatibility
        for images, labels in torch.utils.data.DataLoader(dataset=test_data, batch_size=params.test_batch_size):
            images, labels = images.cuda(), labels.cuda()
            f, prediction_logits = model.forward(images)
            predictions = prediction_logits.argmax(dim=1).cpu().numpy()
            predicted_labels.extend(predictions)
            true_labels = labels.cpu().numpy()
            actual_labels.extend(true_labels)

    confusion_mat = confusion_matrix(actual_labels, predicted_labels)
    confusion_df = pd.DataFrame(confusion_mat / np.sum(confusion_mat) * 10,
                                index=test_data.classes,
                                columns=test_data.classes)
    plt.figure(figsize=(12, 7))
    sn.heatmap(confusion_df, annot=True)
    plt.savefig(f'{params.checkpoint_dir}/heatmap.png', dpi=400)
    return
