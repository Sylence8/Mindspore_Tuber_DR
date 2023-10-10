import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer
from mindspore.common import dtype as mstype

class Loss(nn.Cell):
    def __init__(self, num_hard=0):
        super(Loss, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])
        cls = self.classify_loss(outs, labels)
        return cls, cls

class AUCPLoss(nn.Cell):
    def __init__(self, num_hard=0, lamb=2, alpha=0.5):
        super(AUCPLoss, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard
        self.lamb = lamb
        self.alpha = alpha

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])
        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0
        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = ops.ReduceMean()(ops.Pow(1 - (trans_pos - trans_neg), self.lamb)) / self.lamb
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = ops.ReduceMean()(ops.Pow(1 - (trans_pos - trans_neg), self.lamb)) / self.lamb
            else:
                trans_pos = ops.Tile()(out_pos, (num_neg,))
                trans_neg = ops.Tile()(ops.Transpose(out_neg, (1, 0)), (1, num_pos))
                penalty_term = ops.ReduceMean()(ops.Pow(1 - (trans_pos - trans_neg), self.lamb)) / self.lamb
        except:
            raise ValueError("Error in penalty term calculation")

        cls = self.classify_loss(outs, labels) + self.alpha * penalty_term
        return cls, self.alpha * penalty_term

class PAUCPLoss(nn.Cell):
    def __init__(self, num_hard=0, lamb=2, alpha=0.5):
        super(PAUCPLoss, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard
        self.lamb = 2
        self.alpha = 1

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])
        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0
        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = ops.ReduceMean()(ops.Pow(1 - (trans_pos - trans_neg), self.lamb)) / self.lamb
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = ops.ReduceMean()(ops.Pow(1 - (trans_pos - trans_neg), self.lamb)) / self.lamb
            else:
                trans_pos = ops.Tile()(out_pos, (num_neg,))
                trans_neg = ops.Tile()(ops.Transpose(out_neg, (1, 0)), (1, num_pos))
                penalty_term = ops.ReduceMean()(ops.Pow(1 - (trans_pos - trans_neg), self.lamb)) / self.lamb
        except:
            raise ValueError("Error in penalty term calculation")

        cls = self.alpha * penalty_term
        return cls, self.alpha * penalty_term

class AUCHLoss(nn.Cell):
    def __init__(self, num_hard=0):
        super(AUCHLoss, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])
        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0
        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = ops.ReduceMean()(1 - (trans_pos - trans_neg))
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = ops.ReduceMean()(1 - (trans_pos - trans_neg))
            else:
                trans_pos = ops.Tile()(out_pos, (num_neg,))
                trans_neg = ops.Tile()(ops.Transpose(out_neg, (1, 0)), (1, num_pos))
                penalty_term = ops.ReduceMean()(1 - (trans_pos - trans_neg))
        except:
            raise ValueError("Error in penalty term calculation")

        cls = self.classify_loss(outs, labels) + 0.1 * penalty_term
        return cls, 0.1 * penalty_term

class PAUCLoss(nn.Cell):
    def __init__(self, num_hard=0):
        super(PAUCLoss, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])
        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0
        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = ops.ReduceMean()(1 - (trans_pos - trans_neg))
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = ops.ReduceMean()(1 - (trans_pos - trans_neg))
            else:
                trans_pos = ops.Tile()(out_pos, (num_neg,))
                trans_neg = ops.Tile()(ops.Transpose(out_neg, (1, 0)), (1, num_pos))
                penalty_term = ops.ReduceMean()(1 - (trans_pos - trans_neg))
        except:
            raise ValueError("Error in penalty term calculation")

        cls = penalty_term
        return cls, 0.1 * penalty_term

class FPLoss(nn.Cell):
    def __init__(self, num_hard=0):
        super(FPLoss, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.classify_loss = nn.BCELoss()

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])

        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:, :1])

        pos_loss = ops.Mul()(labels, ops.Log()(outs))
        neg_loss = ops.Mul()(neg_labels, ops.Log()(neg_outs))

        h_pos_loss = ops.Mul()(neg_outs, pos_loss)
        h_neg_loss = ops.Mul()(outs, neg_loss)

        fpcls = - ops.ReduceMean()(h_pos_loss) - ops.ReduceMean()(h_neg_loss)

        return fpcls

class FPLoss1(nn.Cell):
    def __init__(self, num_hard=0):
        super(FPLoss1, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.classify_loss = nn.BCELoss()

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])

        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:, :1])

        pos_loss = ops.Mul()(labels, ops.Log()(outs))
        neg_loss = ops.Mul()(neg_labels, ops.Log()(neg_outs))

        h_pos_loss = ops.Mul()(neg_outs, pos_loss)
        h_neg_loss = ops.Mul()(outs, neg_loss)

        fpcls = - ops.ReduceMean()(h_pos_loss) - 2 * ops.ReduceMean()(h_neg_loss)

        return fpcls

class CELoss(nn.Cell):
    def __init__(self, num_hard=0):
        super(CELoss, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.classify_loss = nn.BCELoss()

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])

        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:, :1])

        pos_loss = ops.Mul()(labels, ops.Log()(outs))
        neg_loss = ops.Mul()(neg_labels, ops.Log()(neg_outs))

        fpcls = - ops.ReduceMean()(pos_loss) - ops.ReduceMean()(neg_loss)

        return fpcls, fpcls

class CWCELoss(nn.Cell):
    def __init__(self, num_hard=0):
        super(CWCELoss, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.classify_loss = nn.BCELoss()

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])

        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:, :1])

        num_neg = ops.ReduceSum()(neg_labels)
        num_pos = ops.ReduceSum()(labels)

        Beta_P = num_pos / (num_pos + num_neg)
        Beta_N = num_neg / (num_pos + num_neg)

        pos_loss = ops.Mul()(labels, ops.Log()(outs))
        neg_loss = ops.Mul()(neg_labels, ops.Log()(neg_outs))

        fpcls = - Beta_N * ops.ReduceMean()(pos_loss) - Beta_P * ops.ReduceMean()(neg_loss)

        return fpcls, fpcls

class RCLoss(nn.Cell):
    def __init__(self, num_hard=0):
        super(RCLoss, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.classify_loss = nn.BCELoss()

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])

        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:, :1])

        pos_loss = ops.Mul()(labels, ops.Log()(outs))
        neg_loss = ops.Mul()(neg_labels, ops.Log()(neg_outs))

        h_pos_loss = ops.Mul()(neg_outs, pos_loss)
        h_neg_loss = ops.Mul()(outs, neg_loss)

        fpcls = - 2 * ops.ReduceMean()(h_pos_loss) - ops.ReduceMean()(h_neg_loss)

        return fpcls

class FPSimilarityLoss(nn.Cell):
    def __init__(self, num_hard=0):
        super(FPSimilarityLoss, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.classify_loss = nn.BCELoss()

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])

        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:, :1])

        pos_loss = ops.Mul()(labels, ops.Log()(outs))
        neg_loss = ops.Mul()(neg_labels, ops.Log()(neg_outs))

        h_pos_loss = ops.Mul()(neg_outs, pos_loss)
        h_neg_loss = ops.Mul()(outs, neg_loss)

        fpcls = - h_pos_loss.mean() - 2 * h_neg_loss.mean()

        return fpcls

class MultiTaskLoss(nn.Cell):
    def __init__(self, alpha=0.5, beta=0.5, pos_weight=None):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.sigmoid = ops.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.cel_loss = nn.CrossEntropyLoss()
        self.bce_logit_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        l_num = output.shape[1] - 12
        if l_num == 1:
            outs = self.sigmoid(output[:, :l_num])
            neg_labels = 1 - labels[:, :l_num]
            neg_outs = 1 - self.sigmoid(output[:, :l_num])
            pos_loss = ops.Mul()(labels[:, :l_num], ops.Log()(outs))
            neg_loss = ops.Mul()(neg_labels, ops.Log()(neg_outs))
            h_pos_loss = ops.Mul()(neg_outs, pos_loss)
            h_neg_loss = ops.Mul()(outs, neg_loss)
            fpcls = - ops.ReduceMean()(h_pos_loss) - ops.ReduceMean()(h_neg_loss)
        else:   
            outs = output[:, :l_num]
            target = labels[:, 0]
            fpcls = self.cel_loss(outs, target.astype(mstype.int32))
        
        char_outs = self.sigmoid(output[:, l_num:])
        char_labels = labels[:, 1:]
            
        pos_weight = Tensor([14.7, 5.2, 6.6, 3.0, 86.1, 169.4, 22.1, 403.8, 145.8, 3.7, 1750, 25], mstype.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        char_loss = criterion(char_outs, char_labels)
        
        mtcls = self.alpha * fpcls + self.beta * char_loss
        
        return mtcls
