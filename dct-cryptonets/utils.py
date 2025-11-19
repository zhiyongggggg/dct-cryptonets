""" Utility classes and functions """

import random
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


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
        device = next(self.parameters()).device
        x = Variable(x.to(device))
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return out, scores
    
    def forward_loss(self, x, y):
        scores = self.forward(x)
        device = next(self.parameters()).device
        y = Variable(y.to(device))
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
        return -1


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


class MetricsTracker(object):
    """Tracks classification metrics including precision, recall, and F1"""
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, preds, targets):
        """
        Update with batch predictions and targets
        Args:
            preds: tensor of predictions (batch_size,)
            targets: tensor of ground truth labels (batch_size,)
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        self.predictions.extend(preds.tolist())
        self.targets.extend(targets.tolist())
    
    def compute_metrics(self, average='binary'):
        """
        Compute precision, recall, and F1 score
        Args:
            average: 'binary' for binary classification, 'macro' or 'weighted' for multi-class
        Returns:
            dict with precision, recall, f1_score
        """
        if len(self.predictions) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # For binary classification (deepfake detection)
        precision = precision_score(targets, predictions, average=average, zero_division=0)
        recall = recall_score(targets, predictions, average=average, zero_division=0)
        f1 = f1_score(targets, predictions, average=average, zero_division=0)
        
        return {
            'precision': precision * 100,  # Convert to percentage
            'recall': recall * 100,
            'f1_score': f1 * 100
        }


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
    num_classes = output.size(1)
    
    maxk = min(maxk, num_classes)
    topk = tuple(min(k, num_classes) for k in topk)

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
    with torch.no_grad():
        for images, labels in torch.utils.data.DataLoader(dataset=test_data, batch_size=params.test_batch_size):
            device = next(model.parameters()).device
            images, labels = images.to(device), labels.to(device)
            
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