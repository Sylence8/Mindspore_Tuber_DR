import csv
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_acc_chara(outputs, targets):
    batch_size = targets.size(0)
    n_classes = outputs.shape[1]
    #print(n_classes)
    macro_acc = np.zeros((batch_size,))
    if n_classes == 3:
        onehot = np.zeros((batch_size,3))
        for m in range(batch_size):
            #print(targets.data.cpu().numpy()[m])
            #ind = 
            onehot[m][int(targets.data.cpu().numpy()[m][0])] = 1
        acc = calculate_accuracy(outputs,targets.long())
        macro_acc = acc
    elif n_classes == 12:
        for i in range(n_classes):
            #print(targets[:,i])
            acc = acc_metric(outputs[:,i],targets[:,i])
            macro_acc[i] = acc
    return np.mean(macro_acc)

def acc_metric(pred,labels):
    bsize = pred.shape[0]
    pred_ = pred > 0
    #print(pred_,labels)
    acc = np.sum(pred_.data.cpu().numpy() == labels.data.cpu().numpy())
    # import pdb;pdb.set_trace()
    acc = acc * 1.0 / bsize
    return acc

def calculate_accuracy(outputs, targets):
    batch_size = targets.shape[0]
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / batch_size
