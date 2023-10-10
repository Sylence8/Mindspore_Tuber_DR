#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import mindspore
from mindspore import context, Tensor
from mindspore.nn import DataParallel
from mindspore.train import Model
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.dataset import DataLoader
from mindspore import nn
from mindspore.common import set_seed
from mindspore.ops import operations as P
import os
from mindspore.train.serialization import load_checkpoint, save_checkpoint, export



from model import generate_model, generate_cammodel
from opts import parse_opts
from layers import *
from metrics import *
from DataIter import MBDataIterSensi_Resis, MBDataIterResisClassify
from tensorboardX import SummaryWriter
from utils import AverageMeter, calculate_accuracy, calculate_acc_chara
from transform import Resize

def train(model, data_loader, optimizer, loss, epoch):
    train_loss = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (G, target, names) in enumerate(data_loader):
        data = mindspore.Tensor(G, mindspore.float32)
        data = Resize([opt.sample_duration, opt.sample_size, opt.sample_size])(data)
        target = mindspore.Tensor(target, mindspore.float32)
        out = model(data)

        if "FP" in opt.save_dir:
            cls = loss(out, target)
        elif "BCE" in opt.save_dir:
            cls = loss(out, target)
        elif "CEL" in opt.save_dir:
            cls = loss(out, target.astype(mindspore.int32))
        else:
            cls = loss(out, target)

        optimizer.clear_gradients()
        cls.backward()
        optimizer.step()

        pred = nn.Sigmoid()(out[:, :1])
        if opt.n_classes == 1:
            train_acc = acc_metric(pred.asnumpy(), target.asnumpy())
        elif "character" in opt.valid_path:
            if "classify3" not in opt.valid_path:
                train_acc_binary = acc_metric(pred.asnumpy(), target.asnumpy())
                train_acc_character = calculate_acc_chara(out[:, opt.n_classes-12:], target[:, 1:])
            else:
                train_acc_binary = calculate_acc_chara(out[:, :opt.n_classes-12], target[:, :1])
                train_acc_character = calculate_acc_chara(out[:, opt.n_classes-12:], target[:, 1:])
        else:
            train_acc = calculate_accuracy(out, target.astype(mindspore.int32))

        try:
            train_loss.append(cls.asnumpy()[0])
        except:
            train_loss.append(cls.asnumpy().item())

        if i % 5 == 0:
            if opt.n_classes == 1 or opt.n_classes == 3:
                try:
                    print("Training: Epoch %d: %dth batch, loss %2.4f, acc %2.4f, lr: %2.6f!" % (
                    epoch, i, cls.asnumpy().item(), train_acc, lr))
                except:
                    print("Training: Epoch %d: %dth batch, loss %2.4f, acc %2.4f, lr: %2.6f!" % (
                    epoch, i, cls.asnumpy().item(), train_acc, lr))
            else:
                try:
                    print("Training: Epoch %d: %dth batch, loss %2.4f, acc bi %2.4f, acc subtype %2.4f , lr: %2.6f!" % (
                        epoch, i, cls.asnumpy().item(), train_acc_binary, train_acc_character, lr))
                except:
                    print("Training: Epoch %d: %dth batch, loss %2.4f, acc bi %2.4f, acc subtype %2.4f, lr: %2.6f!" % (
                        epoch, i, cls.asnumpy().item(), train_acc_binary, train_acc_character, lr))

    return np.mean(train_loss)


def test(model, data_loader, ckpt_path):
    print('test')
    test_acc = []
    loss_lst = []
    pred_lst = []
    label_lst = []
    prob_lst = []
    label_arr = []
    pred_target_dict = {}

    model.set_train(False)
    for i, (data, target, names) in enumerate(data_loader):
        data = mindspore.Tensor(data, mindspore.float32)
        data = Resize([opt.sample_duration, opt.sample_size, opt.sample_size])(data)
        target = mindspore.Tensor(target, mindspore.float32)
        out = model(data)
        if 'FP' in opt.save_dir or 'BCE' in opt.save_dir:
            cls = loss(out, target)
            pred = nn.Sigmoid()(out[:, :1])
            pred_arr = pred.asnumpy()
        elif 'character' in opt.valid_path:
            if "classify3" not in opt.valid_path:
                cls = loss(out, target)
                pred_bi = nn.Sigmoid()(out[:, :1])
                pred_ch = nn.Sigmoid()(out[:, 1:])
                pred_arr = np.hstack((pred_bi.asnumpy(), pred_ch.asnumpy()))
                pred_bi_arr = pred_bi.asnumpy()
                pred_ch_arr = pred_ch.asnumpy()
            else:
                cls = loss(out, target)
                pred_bi = nn.Sigmoid()(out[:, :3])
                pred_ch = nn.Sigmoid()(out[:, 3:])
                pred_arr = np.hstack((pred_bi.asnumpy(), pred_ch.asnumpy()))
                pred_bi_arr = pred_bi.asnumpy()
                pred_ch_arr = pred_ch.asnumpy()
        else:
            cls = loss(out, target)
            pred = nn.Sigmoid()(out)
            pred_arr = np.argmax(pred.asnumpy(), axis=1)

        loss_lst.append(cls.asnumpy())

        if opt.n_classes == 3:
            prob_arr = pred.asnumpy()
        elif opt.n_classes == 15:
            prob_arr = pred_bi.asnumpy()

        if "character" not in opt.valid_path:
            label_arr = target.asnumpy()
        else:
            label_arr = target.asnumpy()
            label_bi_arr = label_arr[:, :1]
            label_ch_arr = label_arr[:, 1:]

        if opt.n_classes == 1:
            _acc = acc_metric(pred_arr, label_arr)
        elif 'character' in opt.valid_path:
            if "classify3" not in opt.valid_path:
                _acc_bi = acc_metric(pred_bi.asnumpy(), target[:, :1].asnumpy())
                _acc_ch = calculate_acc_chara(pred_ch, target[:, 1:])
            else:
                _acc_bi = calculate_acc_chara(pred_bi.asnumpy(), target[:, :1].asnumpy())
                _acc_ch = calculate_acc_chara(pred_ch, target[:, 1:])
        else:
            _acc = calculate_accuracy(out, target.astype(mindspore.int32))

        if opt.n_classes == 1:
            for i in range(pred_arr.shape[0]):
                pred_target_dict[names[i]] = [pred_arr[i], label_arr[i]]
        elif "classify3" in opt.valid_path:
            for i in range(pred_arr.shape[0]):
                pred_target_dict[names[i]] = [pred_arr[i], label_arr[i]]

        else:
            for i in range(pred_arr.shape[0]):
                pred_target_dict[names[i]] = [pred_arr[i], label_arr[i], prob_arr[i]]

        if "character" in opt.valid_path:
            pred_lst.append(pred_bi_arr)
            label_lst.append(label_bi_arr)
        else:
            pred_lst.append(pred_arr)
            label_lst.append(label_arr)

        if "classify3" in opt.valid_path:
            prob_lst.append(prob_arr)

        if "character" in opt.valid_path:
            test_acc.append(_acc_bi)
        else:
            test_acc.append(_acc)

    test_loss = np.mean(loss_lst)
    acc = np.mean(test_acc)
    if 'FP' in opt.save_dir:
        label_lst = np.concatenate(label_lst, axis=0)[:, 0].tolist()
        pred_lst = np.concatenate(pred_lst, axis=0)[:, 0].tolist()
    else:
        label_lst = np.concatenate(label_lst, axis=0).tolist()
        pred_lst = np.concatenate(pred_lst, axis=0).tolist()

    if "classify3" in opt.valid_path:
        prob_lst = np.concatenate(prob_lst, axis=0).tolist()
        auc, prec, recall, spec = multiclass_confusion_matrics(label_lst, pred_lst, prob_lst)
    else:
        auc, prec, recall, spec = confusion_matrics(label_lst, pred_lst)
    f1_score = 2 * (prec * recall) / (prec + recall)

    print("Testing: model %s,loss %2.4f, acc %2.4f, auc %2.4f,precision %2.4f,recall %2.4f,specificity %2.4f!" % (
                ckpt_path, test_loss, acc, auc, prec, recall, spec))
    return pred_target_dict, acc, auc, prec, recall, spec, f1_score

def val(model, data_loader, loss, epoch, lr, max_acc, max_auc, acc_max, auc_max, save_recall, save_prec, save_spec, save_f1score):
    test_acc = []
    loss_lst = []

    pred_lst = []
    label_lst = []
    prob_arr = []
    prob_lst = []
    isave_acc = False
    isave_auc = False
    pred_target_dict = {}

    for i, (data, target, names) in enumerate(data_loader):
        data = mindspore.Tensor(data, mindspore.float32)
        data = Resize([opt.sample_duration, opt.sample_size, opt.sample_size])(data)
        target = mindspore.Tensor(target, mindspore.float32)

        out = model(data)
        
        if "FP" in opt.save_dir:
            cls = loss(out, target)
            pred = nn.Sigmoid()(out[:, :1])
            pred_arr = pred.asnumpy()
        elif "BCE" in opt.save_dir:
            cls = loss(out, target)
            pred = nn.Sigmoid()(out[:, :1])
            pred_arr = pred.asnumpy()
        elif 'character' in opt.valid_path or opt.n_classes == 15:
            cls = loss(out, target)
        else:
            cls = loss(out, target)
            pred = nn.Sigmoid()(out)
            pred_arr = np.argmax(pred.asnumpy(), axis=1)

        loss_lst.append(cls.asnumpy())

        if opt.n_classes == 15:
            prob_arr = pred.asnumpy()
        elif opt.n_classes != 1 and 'character' not in opt.valid_path:
            prob_arr = pred.asnumpy()

        if "character" not in opt.valid_path:
            label_arr = target.asnumpy()
        else:
            label_arr = target.asnumpy()
            label_bi_arr = target.asnumpy()[:, :1]
            label_ch_arr = target.asnumpy()[:, 1:]

        if opt.n_classes == 1:
            _acc = acc_metric(pred_arr, label_arr)
        elif opt.n_classes == 3:
            _acc = calculate_accuracy(out, target.astype(mindspore.int32))
        elif "character" in opt.valid_path:
            if "classify3" not in opt.valid_path:
                _acc_bi = acc_metric(pred[:, :1].asnumpy(), target[:, :1].asnumpy())
                _acc_ch = calculate_acc_chara(out[:, opt.n_classes-12:], target[:, 1:])
            else:
                _acc_bi = calculate_acc_chara(pred_bi, target[:, :1])
                _acc_ch = calculate_acc_chara(out[:, 3:], target[:, 1:])
        
        if opt.n_classes == 1:
            for i in range(pred_arr.shape[0]):
                pred_target_dict[names[i]] = [pred_arr[i], label_arr[i]]
        elif "character" in opt.valid_path:
            if opt.n_classes == 13:
                for i in range(pred_arr.shape[0]):
                    pred_target_dict[names[i]] = [pred_arr[i], label_arr[i]]
            else:
                for i in range(pred_arr.shape[0]):
                    pred_target_dict[names[i]] = [pred_arr[i], label_arr[i], prob_arr[i]]
        elif opt.n_classes == 3:
            pred_target_dict[names[i]] = [pred_arr[i], label_arr[i], prob_arr[i]]

        if "character" in opt.valid_path:
            if opt.n_classes == 15:
                pred_lst.append(label_bi_arr)
                label_lst.append(label_bi_arr)
            else:
                pred_lst.append(label_bi_arr)
                label_lst.append(label_bi_arr)
        else:
            pred_lst.append(pred_arr)
            label_lst.append(label_arr)

        if opt.n_classes == 3 or opt.n_classes == 15:
            prob_lst.append(prob_arr)

        if "character" in opt.valid_path:
            test_acc.append(_acc_bi)
        else:
            test_acc.append(_acc)

    test_loss = np.mean(loss_lst)
    acc = np.mean(test_acc)

    if 'FP' in opt.save_dir:
        label_lst = np.concatenate(label_lst, axis=0)[:, 0].tolist()
        pred_lst = np.concatenate(pred_lst, axis=0)[:, 0].tolist()
    else:
        label_lst = np.concatenate(label_lst, axis=0).tolist()
        pred_lst = np.concatenate(pred_lst, axis=0).tolist()

    if "classify3" in opt.valid_path:
        prob_lst = np.concatenate(prob_lst, axis=0).tolist()
        auc, prec, recall, spec = multiclass_confusion_matrics(label_lst, pred_lst, prob_lst)
    else:
        auc, prec, recall, spec = confusion_matrics(label_lst, pred_lst)
    f1_score = 2 * (prec * recall) / (prec + recall)

    if acc > max_acc:
        max_acc = acc
        max_auc = auc
        save_recall = recall
        save_prec = prec
        save_spec = spec
        save_f1score = f1_score
        isave_acc = True

    if auc > auc_max:
        auc_max = auc
        acc_max = acc
        save_recall = recall
        save_prec = prec
        save_spec = spec
        save_f1score = f1_score
        isave_auc = True
    
    if opt.n_classes == 1 or opt.n_classes == 3:
        print("Validating: Epoch %d:%dth batch, learning rate %2.6f loss %2.4f, acc %2.4f, auc %2.4f, precision %2.4f, recall %2.4f, specificity %2.4f!" % (
            epoch, i, lr, test_loss, acc, auc, prec, recall, spec))
    else:
        print("Validating: Epoch %d:%dth batch, learning rate %2.6f loss %2.4f, acc %2.4f, acc bi %2.4f, acc ch %2.4f, auc %2.4f, precision %2.4f, recall %2.4f, specificity %2.4f!" % (
            epoch, i, lr, test_loss, acc, _acc_bi, _acc_ch, auc, prec, recall, spec))
    
    return max_acc, max_auc, acc_max, auc_max, test_loss, isave_acc, pred_target_dict, isave_auc, save_recall, save_prec, save_spec, save_f1score

icontext.set_context(mode=context.GRAPH_MODE, device_target="GPU")

if __name__ == '__main__':
    # Initialize the opts
    opt = parse_opts()

    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)

    opt.scales = [opt.initial_scale]

    # construct training data iterator
    if opt.task == "sensi":
        train_iter = MBDataIterSensi_Resis(
            data_file=opt.valid_path + '/train_%d.npy' % opt.num_valid,
            data_dir=opt.data_dir,
            phase='train',
            crop_size=opt.crop_size,
            crop_depth=opt.sample_duration,
            sample_size=opt.sample_size,
            sample_phase=None)
    elif opt.task == "resis":
        train_iter = MBDataIterResisClassify(
            data_file=opt.valid_path + '/train_%d.npy' % opt.num_valid,
            data_dir=opt.data_dir,
            phase='train',
            crop_size=opt.crop_size,
            crop_depth=opt.sample_duration,
            sample_size=opt.sample_size,
            sample_phase=None)

    train_loader = DataLoader(
        dataset=train_iter,
        batch_size=opt.batch_size,
        shuffle=True,
        num_parallel_workers=0,
        drop_last=True)

    if opt.task == "sensi":
        val_iter = MBDataIterSensi_Resis(data_file=opt.valid_path + '/val_%d.npy' % opt.num_valid, data_dir=opt.data_dir,
                                         phase='test', crop_size=opt.crop_size,
                                         crop_depth=opt.sample_duration, sample_size=opt.sample_size)
    elif opt.task == "resis":
        val_iter = MBDataIterResisClassify(data_file=opt.valid_path + '/val_%d.npy' % opt.num_valid,
                                           data_dir=opt.data_dir,
                                           phase='test', crop_size=opt.crop_size,
                                           crop_depth=opt.sample_duration, sample_size=opt.sample_size)

    val_loader = DataLoader(
        dataset=val_iter,
        batch_size=opt.batch_size,
        shuffle=True,
        num_parallel_workers=0,
        drop_last=True)

    model, policies = generate_model(opt)
    model = Model(model)

    if opt.n_classes == 1:
        if "FP" in opt.save_dir:
            loss = FPLoss()  # Using FPLoss
        else:
            loss = nn.BCEWithLogitsLoss()
    elif "CEL" in opt.save_dir:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    elif opt.n_classes > 10:
        loss = MultiTaskLoss(alpha=opt.alpha, beta=1 - opt.alpha)

    optimizer = nn.SGD(
        params=model.trainable_params(),
        learning_rate=opt.lr,
        momentum=0.9,
        weight_decay=1e-2)

    max_acc = 0
    max_auc = 0
    acc_max = 0
    auc_max = 0
    save_recall = 0
    save_prec = 0
    save_spec = 0
    save_f1score = 0

    max_dict = {}
    max_auc_dict = {}

    save_dir = "saved_models/" + opt.task + "/class" + str(opt.n_classes) + "/" + opt.save_dir + "/%d" % opt.num_valid
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(opt.start_epoch, opt.epochs):
        train_loss = train(model, train_loader, optimizer, loss, epoch)
        max_acc, max_auc, acc_max, auc_max, val_loss, isave_acc, pred_target_dict, isave_auc, save_recall, save_prec, save_spec, save_f1score = val(
            model, val_loader, loss, epoch, opt.lr, max_acc, max_auc, acc_max, auc_max, save_recall, save_prec, save_spec, save_f1score)
        if isave_acc:
            max_dict = pred_target_dict

        if isave_auc:
            max_auc_dict = pred_target_dict

        if isave_acc or isave_auc:
            save_checkpoint({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': model.state_dict()
            }, os.path.join(save_dir, 'checkpoint_%03d.ckpt' % epoch))

        print("Epoch %d, the max acc is %2.4f, max auc is %2.4f, the acc max is %2.4f, auc max is %2.4f" % (
            epoch, max_acc, max_auc, acc_max, auc_max))

        print('\n')
        if epoch >= 50 and epoch % 30 == 0:
            opt.lr = opt.lr * 0.5
            optimizer.set_learning_rate(opt.lr)
        if opt.task == "sensi":
            train_iter = MBDataIterSensi_Resis(data_file=opt.valid_path + '/train_%d.npy' % opt.num_valid,
                                               data_dir=opt.data_dir, phase='train',
                                               crop_size=opt.crop_size, crop_depth=opt.sample_duration,
                                               sample_size=opt.sample_size, sample_phase=None)
        elif opt.task == "resis":
            train_iter = MBDataIterResisClassify(data_file=opt.valid_path + '/train_%d.npy' % opt.num_valid,
                                                 data_dir=opt.data_dir, phase='train',
                                                 crop_size=opt.crop_size, crop_depth=opt.sample_duration,
                                                 sample_size=opt.sample_size, sample_phase=None)
        train_loader = DataLoader(
            dataset=train_iter,
            batch_size=opt.batch_size,
            shuffle=True,
            num_parallel_workers=0,
            drop_last=True)

    results = {}
    results['max_acc'] = max_acc
    results['max_auc'] = max_auc
    results['acc_max'] = acc_max
    results['auc_max'] = auc_max

    results['recall'] = save_recall
    results['prec'] = save_prec
    results['spec'] = save_spec
    results['f1score'] = save_f1score

    results['max_dict'] = max_dict
    results['max_auc_dict'] = max_auc_dict

    save_results_dir = "results/" + opt.task + "/" + opt.save_dir
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    np.save(save_results_dir + "/valid_%d.npy" % opt.num_valid, results)
    print("The max acc is %2.4f, max auc is %2.4f, acc_max is %2.4f, auc_max is %2.4f" % (
        max_acc, max_auc, acc_max, auc_max))
    print(opt.no_val)

    if not opt.no_val:
        print("test")
        if opt.task == "sensi":
            test_iter = MBDataIterSensi_Resis(data_file=opt.valid_path + '/test_%d.npy' % opt.num_valid,
                                              data_dir=opt.data_dir, phase='test', crop_size=opt.crop_size,
                                              crop_depth=opt.sample_duration, sample_size=opt.sample_size,
                                              sample_phase=None)
        elif opt.task == "resis":
            test_iter = MBDataIterResisClassify(data_file=opt.valid_path + '/test_%d.npy' % opt.num_valid,
                                                data_dir=opt.data_dir, phase='test', crop_size=opt.crop_size,
                                                crop_depth=opt.sample_duration, sample_size=opt.sample_size,
                                                sample_phase=None)
        test_loader = DataLoader(
            dataset=test_iter,
            batch_size=opt.batch_size,
            shuffle=True,
            num_parallel_workers=0,
            drop_last=True)

        model_test, _ = generate_model(opt)
        ckpt = load_checkpoint(os.path.join(save_dir, 'checkpoint_%03d.ckpt' % epoch))
        model_test.load_state_dict(ckpt['state_dict'])
        model_test.set_train(False)
        results_test = {}
        dict_1, acc1, auc1, prec1, recall1, spec1, f1_score1 = test(model_test, test_loader, epoch)
        results_test['max_acc_dict'] = dict_1
        dict_2, acc2, auc2, prec2, recall2, spec2, f1_score2 = test(model_test, test_loader, epoch)
        results_test['max_auc_dict'] = dict_2
        dict_3, acc3, auc3, prec3, recall3, spec3, f1_score3 = test(model_test, test_loader, epoch)
        results_test['max_both_dict'] = dict_3
        np.save(save_results_dir + "/test_%d.npy" % opt.num_valid, results_test)