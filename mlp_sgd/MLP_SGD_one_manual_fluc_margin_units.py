import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Function, Variable
from chainer import datasets, optimizers, serializers
from chainer import Link, Chain, ChainList

import os
import random
import copy
import datetime
import time
import code
import sys
import shutil
import pickle

import json
import collections

from chainer.backends import cuda

from search_grad import *
from dataset import *

############################################################################
seed_num = 2019014
############################################################################


############################################################################
gpu_id = 1
############################################################################
if gpu_id is not None:
    if gpu_id >= 0:
        cuda.get_device_from_id(gpu_id).use()
        xp = cuda.cupy
        cpu = cuda.to_cpu
    else:
        xp = np
        cpu = lambda value: value
    # end if
else:
    xp = np
    cpu = lambda value: value
# end if
############################################################################


############################################################################
out_dir = 'MLP_SGD_One_Manual_Fluc_Margin_Units_'
out_dir += datetime.datetime.today().strftime('%y%m%d%H%M%S')
os.makedirs(out_dir, exist_ok=True)
############################################################################


############################################################################
ds_name = 'MNIST'
# ds_name = 'CIFAR10'
# ds_name = 'CIFAR100'
############################################################################
data_type = 'train'
# data_type = 'test'
############################################################################
pair_type = 'same'
# pair_type = 'diff'
# pair_type = 'rand'
############################################################################
# ds_size = None
ds_size = 28
############################################################################
ds_normalized = True
l2_normalized = True  ## True fixed: Remark: init preserve
############################################################################
if (ds_name == 'MNIST'):
    data_dir = 'data_MNIST'
    num_classes = 10
    in_ch = 1
elif (ds_name == 'CIFAR10'):
    data_dir = 'data_CIFAR10'
    num_classes = 10
    in_ch = 3
elif (ds_name == 'CIFAR100'):
    data_dir = 'data_CIFAR100'
    num_classes = 100
    in_ch = 3
# end if
os.makedirs(data_dir, exist_ok=True)
############################################################################


############################################################################
# MLP networks
############################################################################
model_name = 'MLP'
# num_units = 1000
num_layers = 2
############################################################################
units_list = [1000, 3000, 5000, 7000, 9000]
############################################################################
epoch_list = [2500, 2500, 2500, 2500, 2500]
batch_list = [ 500,  500,  500,  500,  500]
trig_list  = [  20,   20,   20,   20,   20] # epoch trigger
############################################################################


############################################################################
"""
############################################################################
units_list = [100, 200]
epoch_list = [ 50,  50]
batch_list = [500, 500]
trig_list  = [ 10,  10]
############################################################################
"""
############################################################################


############################################################################
# opt = 'Adam'
opt = 'SGD'
# opt = 'MomentumSGD'
# opt = 'NesterovAG'
# opt = 'AdaDelta'
############################################################################
ep_init = 200
lr_init = 0.00001
lr = 0.001
############################################################################
train_bat = 128
############################################################################
test_bat  = 512
############################################################################


############################################################################
## parameter for gap deviation
############################################################################
trgt_num = 128 ## gradients at begin, middle and end points
############################################################################
## parameter for divided interval
############################################################################
max_div  = 1000  ## max divided number
############################################################################


############################################################################
def init_weight(sigma, in_size, out_size):
    return np.random.normal(0.0, sigma, (out_size, in_size))
############################################################################


############################################################################
class MLP(chainer.Chain):
    def __init__(self, nlayers, nunits, nclasses, ds_size, in_ch):
        # hw = chainer.initializers.HeNormal()
        # lw = chainer.initializers.HeNormal(scale=1.0/np.sqrt(2.0))
        # lw = chainer.initializers.LeCunNormal()
        super(MLP, self).__init__()
        with self.init_scope():
            ################################################################
            self._forward = []
            ################################################################
            name = 'l0'
            in_size = ds_size * ds_size * in_ch
            W = init_weight(np.sqrt(2.0/in_size), in_size, nunits)
            proj = L.Linear(in_size, nunits, initialW=W)
            setattr(self, name, proj)
            self._forward.append(name)
            ################################################################
            for i in range(1, nlayers - 1):
                name = 'l{}'.format(i)
                W = init_weight(np.sqrt(2.0/nunits), nunits, nunits)
                proj = L.Linear(nunits, nunits, initialW=W)
                setattr(self, name, proj)
                self._forward.append(name)
            # end for
            ################################################################
            W = init_weight(np.sqrt(2.0/nunits), nunits, nclasses)
            self.fin = L.Linear(nunits, nclasses, initialW=W)
            ################################################################
        # end with
        self.output = None

    def __call__(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
            x = F.relu(x)
        return self.fin(x)

    def get_act(self, x):
        act = []
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
            act.append(x.data > 0)
            x = F.relu(x)
        self.output = (self.fin(x)).data
        return act

    def cutoff(self, x, cut):
        for cid, name in enumerate(self._forward):
            l = getattr(self, name)
            x = l(x) * cut[cid]
        return self.fin(x)
############################################################################


############################################################################
def get_model(num_layers, num_units, num_classes, ds_size, in_ch):
    return MLP(num_layers, num_units, num_classes, ds_size, in_ch)
############################################################################


############################################################################
def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    cuda.cupy.random.seed(seed)
############################################################################


############################################################################
def get_optimizer(opt, lr=0.001):
    ########################################################################
    if (opt == 'Adam'):
        ####################################################################
        optimizer = chainer.optimizers.Adam()
    elif (opt == 'AdaDelta'):
        ####################################################################
        optimizer = chainer.optimizers.AdaDelta()
        ####################################################################
    elif (opt == 'SGD'):
        ####################################################################
        optimizer = chainer.optimizers.SGD(lr=lr)
        ####################################################################
    elif (opt == 'MomentumSGD'):
        ####################################################################
        optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=0.9)
        ####################################################################
    elif (opt == 'NesterovAG'):
        ####################################################################
        optimizer = optimizers.NesterovAG(lr=lr, momentum=0.9)
        ####################################################################
    # end if
    ########################################################################
    return optimizer
############################################################################


############################################################################
def calc_output(model, data_ds):
    ########################################################################
    x_data = []
    for lid in range(len(data_ds)):
        x_data.append(data_ds[lid][0])
    # end for
    x_data = np.asarray(x_data, dtype=np.float32)
    ########################################################################
    # NN's value
    nn = lambda x: model(x)
    ########################################################################
    ## get output values
    output_val = []
    n_sample = x_data.shape[0]
    ########################################################################
    for i in range(0, n_sample, test_bat):
        ####################################################################
        xx = Variable(xp.asarray(x_data[i:i + test_bat].astype(np.float32)))
        ####################################################################
        with chainer.using_config('train', False), \
             chainer.using_config('enable_backprop', False):
            output_val.append(nn(xx).data)
        # end with
        ####################################################################
    # end for
    ########################################################################
    out_arr = xp.concatenate(output_val, axis=0)
    l2_out = cpu(xp.sqrt((out_arr * out_arr).sum(axis=1)))
    ########################################################################
    # NN's value
    nn = lambda x: model.get_act(x)
    ########################################################################
    xx = Variable(xp.asarray(x_data[0:1].astype(np.float32)))
    with chainer.using_config('train', False):
        ro_data = nn(xx)
    # end with
    ########################################################################
    num_acts = 0
    for val in ro_data:
        num_acts += (val[0]).size
    # end for
    ########################################################################
    return l2_out, num_acts
############################################################################


############################################################################
def get_div(num_acts):
    ########################################################################
    if (num_acts > max_div):
        ####################################################################
        x_div = np.linspace(0.0, 1.0, max_div, dtype=np.float)
        ####################################################################
    else:
        ####################################################################
        x_div = np.linspace(0.0, 1.0, num_acts, dtype=np.float)
        ####################################################################
    # end if
    ########################################################################
    return x_div
############################################################################


############################################################################
def calc_margin(marg_list, t_data, sect_ids):
    ########################################################################
    class_arr = np.arange(num_classes, dtype=np.int)
    ########################################################################
    edge_mrg = []
    max_fluc = []
    ########################################################################
    for mid in range(len(sect_ids)):
        ####################################################################
        t_lab = t_data[sect_ids[mid][0]]
        n_lab = t_data[sect_ids[mid][1]]
        arr = (marg_list[mid]).T
        ####################################################################
        tn_bool = (class_arr == t_lab) | (class_arr == n_lab)
        tn_arr = arr[tn_bool]
        ot_arr = arr[~tn_bool]
        ####################################################################
        if (t_lab == n_lab):
            max_tn = tn_arr.reshape((-1))
        else:
            max_tn = tn_arr.max(axis=0).reshape((-1))
        # end if
        ####################################################################
        max_ot = ot_arr.max(axis=0).reshape((-1))
        ####################################################################
        beg_mrg = max_tn[0]  - max_ot[0]
        end_mrg = max_tn[-1] - max_ot[-1]
        edge_mrg.append((beg_mrg + end_mrg) / 2.0)
        ####################################################################
        lin_arr = np.linspace(max_tn[0], max_tn[-1], max_tn.shape[0])
        max_fluc.append(np.abs(max_tn - lin_arr).max())
        ####################################################################
    # end for
    ########################################################################
    return edge_mrg, max_fluc
############################################################################


############################################################################
def calc_fluctuation(model, pair_ids, data_ds, b_data, batch, l2_init):
    ########################################################################
    if (l2_normalized == True):
        l2_out, num_acts = calc_output(model, data_ds)
        l2_out = l2_out / l2_init
    elif(l2_normalized == False):
        l2_out, num_acts = calc_output(model, data_ds)
        l2_out = np.ones(len(data_ds))
    # end if
    ########################################################################
    x_div = get_div(num_acts)
    ########################################################################
    # hess_list, switch_list, g_sig_list, marg_list = bf_new_mrg(
    #     model, data_ds, pair_ids, l2_out, x_div, num_classes, batch)
    hess_list, switch_list, g_sig_list, marg_list = bf_margin(
        model, data_ds, pair_ids, l2_out, x_div, num_classes, num_acts)
    ########################################################################
    ## make t_data: label data
    t_data = []
    for lid in range(len(data_ds)):
        t_data.append(data_ds[lid][1])
    # end for
    t_data = np.asarray(t_data, dtype=np.int32)
    ########################################################################
    edge_mrg, max_fluc = calc_margin(marg_list, t_data, pair_ids)
    ########################################################################
    # for switch mean
    switch_arr  = np.array(switch_list)
    switch_mean = switch_arr.mean()
    switch_std  = switch_arr.std()
    ########################################################################
    ## hess sigma (with amount)
    h_sigma_arr  = np.array(hess_list)
    h_sigma_mean = h_sigma_arr.mean()
    h_sigma_std  = h_sigma_arr.std()
    ########################################################################
    ## hess fluc
    h_fluc_arr  = h_sigma_arr * np.sqrt(switch_mean) / 2
    h_fluc_mean = h_fluc_arr.mean()
    h_fluc_std  = h_fluc_arr.std()
    ########################################################################
    g_sigma_arr  = np.array(g_sig_list)
    g_sigma_std  = g_sigma_arr.std()
    ########################################################################
    ## grad fluc
    g_fluc_arr  = g_sigma_arr * np.sqrt(switch_mean) / 2
    g_fluc_std  = g_fluc_arr.std()
    ########################################################################
    b_data['switch_mean'].append(switch_mean)
    b_data['switch_std'].append(switch_std)
    ########################################################################
    b_data['g_sigma_std'].append(g_sigma_std)
    b_data['g_fluc_std'].append(g_fluc_std)
    ########################################################################
    b_data['h_sigma_mean'].append(h_sigma_mean)
    b_data['h_sigma_std'].append(h_sigma_std)
    b_data['h_fluc_mean'].append(h_fluc_mean)
    b_data['h_fluc_std'].append(h_fluc_std)
    ########################################################################
    edge_mrg_arr = np.asarray(edge_mrg)
    max_fluc_arr = np.asarray(max_fluc)
    ########################################################################
    b_data['edge_mrg_mean'].append(edge_mrg_arr.mean())
    b_data['edge_mrg_std'].append(edge_mrg_arr.std())
    b_data['max_fluc_mean'].append(max_fluc_arr.mean())
    b_data['max_fluc_std'].append(max_fluc_arr.std())
    ########################################################################
    g_val = 'switch:{:.2f}, g_sigma:{:.8f}, g_fluc:{:.8f}'.format(
        switch_mean, g_sigma_std, g_fluc_std)
    print(g_val)
    h_val = 'switch:{:.2f}, h_sigma:{:.8f}, h_fluc:{:.8f}\n'.format(
        switch_mean, h_sigma_mean, h_fluc_mean)
    print(h_val)
    ########################################################################
############################################################################


############################################################################
def train_model(train_bat, lr, epoch, model, data_ds, x_train, t_train,
                x_test, t_test, pair_ids, out_dir, num_trig, b_data,
                ep_list, num_units, batch, l2_init,
                log_name, loss_name, acc_name):
    ########################################################################
    setting = 'Units:{}, '.format(num_units)
    setting += 'bs:{}, lr:{}, all_epoch:{}'.format(train_bat, lr, epoch)
    print(setting)
    ########################################################################
    log_file = out_dir + '/' + log_name
    with open(log_file, 'w') as f:
        f.write(setting + '\n')
    # end with
    ########################################################################
    optimizer = get_optimizer(opt, lr)
    optimizer.setup(model)
    optimizer.use_cleargrads()
    ########################################################################
    print('train model')
    N  = x_train.shape[0]
    NT = x_test.shape[0]
    ########################################################################
    train_loss = []
    train_acc  = []
    test_loss  = []
    test_acc   = []
    ########################################################################
    print('Starting initial stabilaizer: epoch 0 to', min([ep_init, epoch]))
    org_rate = optimizer.lr
    init_rate = lr_init
    optimizer.lr = init_rate
    delta0 = (org_rate - init_rate) / (N * ep_init / train_bat)
    ########################################################################
    for eid in range(min([ep_init, epoch])):
        ####################################################################
        epoch_start_time = time.clock()
        ####################################################################
        ## calc. fluctuation
        ####################################################################
        if ((eid % num_trig) == 0):
            ################################################################
            print('\nUnits:{}, '.format(num_units), end='', flush=True)
            calc_fluctuation(model, pair_ids, data_ds, b_data, batch, l2_init)
            ep_list.append(eid)
            ################################################################
        # end if
        ####################################################################
        perm = np.random.permutation(N)
        sum_loss = 0
        sum_acc = 0
        num = 0
        ####################################################################
        # learning loop
        ####################################################################
        for i in range(0, N, train_bat):
            ################################################################
            chosen_ids = perm[i:i + train_bat]
            x = chainer.Variable(xp.asarray(x_train[chosen_ids]))
            t = chainer.Variable(xp.asarray(t_train[chosen_ids]))
            ################################################################
            y = model(x)
            model_loss = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            model_loss.backward()
            optimizer.update()
            ################################################################
            bat_acc  = float(cpu(F.accuracy(y, t).data))
            bat_loss = float(cpu(model_loss.data))
            sum_acc  += bat_acc * t.data.shape[0]
            sum_loss += bat_loss * t.data.shape[0]
            num += t.data.shape[0]
            ################################################################
            ################################################################
            optimizer.lr += delta0
            ################################################################
            ################################################################
        # end for
        ####################################################################
        loss_val = sum_loss / num
        acc_val  = sum_acc / num
        train_loss.append(loss_val)
        train_acc.append(acc_val)
        log_data = collections.OrderedDict()
        log_data['epoch'] = eid
        log_data['train loss'] = loss_val
        log_data['train acc']  = acc_val
        train_out ='train loss:{:.6f}, train acc:{:.6f}'.format(
            loss_val, acc_val)
        ####################################################################
        # evaluation (validation)
        ####################################################################
        sum_loss = 0
        sum_acc  = 0
        num = 0
        ####################################################################
        for i in range(0, NT, test_bat):
            ################################################################
            x = chainer.Variable(xp.asarray(x_test[i:i + test_bat]))
            t = chainer.Variable(xp.asarray(t_test[i:i + test_bat]))
            ################################################################
            with chainer.using_config('train', False), \
                 chainer.using_config('enable_backprop', False):
                y = model(x)
            ################################################################
            model_loss = F.softmax_cross_entropy(y, t)
            bat_acc  = float(cpu(F.accuracy(y, t).data))
            bat_loss = float(cpu(model_loss.data))
            sum_acc  += bat_acc  * t.data.shape[0]
            sum_loss += bat_loss * t.data.shape[0]
            num += t.data.shape[0]
            ################################################################
        # end for
        ####################################################################
        epoch_end_time = time.clock()
        ep_time ='time:{:.3f}'.format(epoch_end_time - epoch_start_time)
        ####################################################################
        lrate = 'lr:{:.5f}'.format(optimizer.lr)
        ####################################################################
        loss_val = sum_loss / num
        acc_val  = sum_acc / num
        test_loss.append(loss_val)
        test_acc.append(acc_val)
        log_data['test loss'] = loss_val
        log_data['test acc']  = acc_val
        test_out = 'test loss:{:.6f}, test acc:{:.4f}'.format(
            loss_val, acc_val)
        print_out = 'epoch:{:04d}, '.format(eid)
        print_out += (train_out + ', ' + test_out + ', ' + ep_time)
        print_out += ', ' + lrate
        print(print_out)
        ####################################################################
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
        # end with
        ####################################################################
    # end for
    ########################################################################
    ## end of initialization epoch
    ########################################################################
    optimizer.lr = org_rate
    ########################################################################
    ## start normal learning epoch
    ########################################################################
    for eid in range(ep_init, max([ep_init, epoch])):
        ####################################################################
        epoch_start_time = time.clock()
        ####################################################################
        ## calc. fluctuation
        ####################################################################
        if ((eid % num_trig) == 0):
            ################################################################
            print('\nUnits:{}, '.format(num_units), end='', flush=True)
            calc_fluctuation(model, pair_ids, data_ds, b_data, batch, l2_init)
            ep_list.append(eid)
            ################################################################
        # end if
        ####################################################################
        perm = np.random.permutation(N)
        sum_loss = 0
        sum_acc  = 0
        num = 0
        ####################################################################
        # learning loop
        ####################################################################
        for i in range(0, N, train_bat):
            ################################################################
            chosen_ids = perm[i:i + train_bat]
            x = chainer.Variable(xp.asarray(x_train[chosen_ids]))
            t = chainer.Variable(xp.asarray(t_train[chosen_ids]))
            ################################################################
            y = model(x)
            model_loss = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            model_loss.backward()
            optimizer.update()
            ################################################################
            bat_acc  = float(cpu(F.accuracy(y, t).data))
            bat_loss = float(cpu(model_loss.data))
            sum_acc  += bat_acc * t.data.shape[0]
            sum_loss += bat_loss * t.data.shape[0]
            num += t.data.shape[0]
            ################################################################
        # end for
        ####################################################################
        loss_val = sum_loss / num
        acc_val  = sum_acc / num
        train_loss.append(loss_val)
        train_acc.append(acc_val)
        log_data = collections.OrderedDict()
        log_data['epoch'] = eid
        log_data['train loss'] = loss_val
        log_data['train acc']  = acc_val
        train_out ='train loss:{:.6f}, train acc:{:.6f}'.format(
            loss_val, acc_val)
        ####################################################################
        # evaluation (validation)
        ####################################################################
        sum_loss = 0
        sum_acc  = 0
        num = 0
        ####################################################################
        for i in range(0, NT, test_bat):
            ################################################################
            x = chainer.Variable(xp.asarray(x_test[i:i + test_bat]))
            t = chainer.Variable(xp.asarray(t_test[i:i + test_bat]))
            ################################################################
            with chainer.using_config('train', False), \
                 chainer.using_config('enable_backprop', False):
                y = model(x)
            ################################################################
            model_loss = F.softmax_cross_entropy(y, t)
            bat_acc  = float(cpu(F.accuracy(y, t).data))
            bat_loss = float(cpu(model_loss.data))
            sum_acc  += bat_acc  * t.data.shape[0]
            sum_loss += bat_loss * t.data.shape[0]
            num += t.data.shape[0]
            ################################################################
        # end for
        ####################################################################
        epoch_end_time = time.clock()
        ep_time ='epoch time:{:.3f}'.format(epoch_end_time - epoch_start_time)
        ####################################################################
        lrate = 'lr:{:.5f}'.format(optimizer.lr)
        ####################################################################
        loss_val = sum_loss / num
        acc_val  = sum_acc / num
        test_loss.append(loss_val)
        test_acc.append(acc_val)
        ####################################################################
        log_data['test loss'] = loss_val
        log_data['test acc']  = acc_val
        test_out = 'test loss:{:.6f}, test acc:{:.4f}'.format(
            loss_val, acc_val)
        print_out = 'epoch:{:04d}, '.format(eid)
        print_out += (train_out + ', ' + test_out + ', ' + ep_time)
        print_out += ', ' + lrate
        print(print_out)
        ####################################################################
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
        # end with
        ####################################################################
        plot_epoch(eid, train_loss, test_loss, out_dir, loss_name, 'loss')
        plot_epoch(eid, train_acc, test_acc, out_dir, acc_name, 'accuracy')
        ####################################################################
    # end for
    ########################################################################
    save_name = '_U{}.npy'.format(num_units)
    np.save(out_dir + '/train_loss' + save_name, np.asarray(train_loss))
    np.save(out_dir + '/test_loss' + save_name, np.asarray(test_loss))
    np.save(out_dir + '/train_acc' + save_name, np.asarray(train_acc))
    np.save(out_dir + '/test_acc' + save_name, np.asarray(test_acc))
    ########################################################################
    if gpu_id is not None:
        if gpu_id >= 0:
            model.to_cpu()
    ########################################################################
############################################################################


############################################################################
def plot_epoch(eid, train_data, test_data, out_dir, file_name, data_type):
    ########################################################################
    f_name = '{}/{}.png'.format(out_dir, file_name)
    ########################################################################
    ep_list = list(range(eid+1))
    ########################################################################
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y1_max = np.asarray(train_data).max()
    y2_max = np.asarray(test_data).max()
    y_max = max([y1_max, y2_max]) * 1.1
    y1_min = np.asarray(train_data).min()
    y2_min = np.asarray(test_data).min()
    y_min = min([y1_min, y2_min]) * 0.9
    ax.plot(ep_list, train_data, label='train')
    ax.plot(ep_list, test_data, label='test')
    ax.set_xlabel('epoch')
    ax.set_ylabel(data_type)
    ax.set_ylim([y_min, y_max])
    ax.set_yscale('log')
    ax.grid(True)
    ax.xaxis.grid(True, which='major', linestyle='-', color='#CFCFCF')
    ax.yaxis.grid(True, which='major', linestyle='-', color='#CFCFCF')
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title(f_name)
    plt.tight_layout()
    plt.savefig(f_name)
    ax.clear()
    fig.clf()
    plt.close(fig)
############################################################################


############################################################################
def save_files():
    ########################################################################
    my_file = os.path.basename(__file__)
    shutil.copy('./' + my_file, './{}/'.format(out_dir) + my_file)
    cg_file = 'search_grad.py'
    shutil.copy('./' + cg_file, './{}/'.format(out_dir) + cg_file)
    ds_file = 'dataset.py'
    shutil.copy('./' + ds_file, './{}/'.format(out_dir) + ds_file)
    ########################################################################
############################################################################


############################################################################
def make_data_dict():
    ########################################################################
    d_dict = {}
    ########################################################################
    d_dict['switch_mean']  = []
    d_dict['switch_std']   = []
    ########################################################################
    d_dict['g_sigma_std']   = []
    ########################################################################
    d_dict['g_fluc_std']    = []
    ########################################################################
    d_dict['h_sigma_mean'] = []
    d_dict['h_sigma_std']  = []
    ########################################################################
    d_dict['h_fluc_mean']  = []
    d_dict['h_fluc_std']   = []
    ########################################################################
    d_dict['edge_mrg_mean'] = []
    d_dict['edge_mrg_std']  = []
    d_dict['max_fluc_mean'] = []
    d_dict['max_fluc_std']  = []
    ########################################################################
    d_dict['ep_list']      = None
    ########################################################################
    return d_dict
############################################################################


############################################################################
def plot_diff(ep_list, g_list, h_list, h_err,
              out_dir, data_name, num_units):
    ########################################################################
    if (data_name != ''):
        su_dt = data_name + '_'
        co_dt = data_name + ' '
    else:
        su_dt = ''
        co_dt = ''
    # end if
    ########################################################################
    model_desc = 'units {}, '.format(num_units) + co_dt + 'data'
    ########################################################################
    filename = '{}/U{}_{}sgd_diff.png'.format(out_dir, num_units, su_dt)
    ########################################################################
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ########################################################################
    g_mean_arr = np.asarray(g_list)
    h_mean_arr = np.asarray(h_list)
    h_std_arr  = np.asarray(h_err)
    g_max = g_mean_arr.max()
    g_min = g_mean_arr.min()
    h_max = (h_mean_arr + h_std_arr).max()
    h_min = (h_mean_arr - h_std_arr).min()
    y_max = max([g_max, h_max]) * 1.1
    m_min = min([g_min, h_min])
    if (m_min < 0.0):
        y_min = m_min * 1.1
    else:
        y_min = m_min * 0.9
    # end if
    ########################################################################
    ax.plot(ep_list, g_list, marker='o', markersize=3,
            alpha=0.6, label='grad. {}'.format(data_name))
    ax.errorbar(ep_list, h_list, yerr=h_err, marker='o', markersize=3,
                alpha=0.6, label='hess. {}'.format(data_name),
                capthick=1, capsize=2, lw=1)
    ax.set_xlabel('epoch')
    ax.set_ylabel(data_name)
    ax.set_ylim([y_min, y_max])
    ax.grid(True)
    ax.xaxis.grid(True, which='major', linestyle='-', color='#CFCFCF')
    ax.yaxis.grid(True, which='major', linestyle='-', color='#CFCFCF')
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title("{}\n{}".format(filename, model_desc))
    plt.tight_layout()
    plt.savefig(filename)
    ########################################################################
    ax.clear()
    fig.clf()
    plt.close(fig)
    ########################################################################
############################################################################


############################################################################
def plot_err(ep_list, d_list, err_list, out_dir, data_name, num_units):
    ########################################################################
    if (data_name != ''):
        su_dt = data_name + '_'
        co_dt = data_name + ' '
    else:
        su_dt = ''
        co_dt = ''
    # end if
    ########################################################################
    model_desc = 'units {}, '.format(num_units) + co_dt + 'data'
    ########################################################################
    filename = '{}/U{}_{}sgd_err.png'.format(out_dir, num_units, su_dt)
    ########################################################################
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ########################################################################
    mean_arr = np.asarray(d_list)
    std_arr  = np.asarray(err_list)
    y_max = (mean_arr + std_arr).max() * 1.1
    m_min = (mean_arr - std_arr).min()
    if (m_min < 0.0):
        y_min = m_min * 1.1
    else:
        y_min = m_min * 0.9
    # end if
    ########################################################################
    ax.errorbar(ep_list, d_list, yerr=err_list, marker='o', markersize=3,
                alpha=0.6, label=data_name, capthick=1, capsize=2, lw=1)
    ax.set_xlabel('epoch')
    ax.set_ylabel(data_name)
    ax.set_ylim([y_min, y_max])
    ax.grid(True)
    ax.xaxis.grid(True, which='major', linestyle='-', color='#CFCFCF')
    ax.yaxis.grid(True, which='major', linestyle='-', color='#CFCFCF')
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title("{}\n{}".format(filename, model_desc))
    plt.tight_layout()
    plt.savefig(filename)
    ########################################################################
    ax.clear()
    fig.clf()
    plt.close(fig)
    ########################################################################
############################################################################


############################################################################
def plot_data(ep_list, d_list, out_dir, data_name, num_units):
    ########################################################################
    if (data_name != ''):
        su_dt = data_name + '_'
        co_dt = data_name + ' '
    else:
        su_dt = ''
        co_dt = ''
    # end if
    ########################################################################
    model_desc = 'units {}, '.format(num_units) + co_dt
    ########################################################################
    filename = '{}/U{}_{}sgd_single.png'.format(out_dir, num_units, su_dt)
    ########################################################################
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ########################################################################
    mean_arr = np.asarray(d_list)
    y_max = mean_arr.max() * 1.1
    m_min = mean_arr.min()
    if (m_min < 0.0):
        y_min = m_min * 1.1
    else:
        y_min = m_min * 0.9
    # end if
    ########################################################################
    ax.plot(ep_list, d_list, marker='o', markersize=3,
            alpha=0.6, label=data_name)
    ax.set_xlabel('epoch')
    ax.set_ylabel(data_name)
    ax.set_ylim([y_min, y_max])
    ax.grid(True)
    ax.xaxis.grid(True, which='major', linestyle='-', color='#CFCFCF')
    ax.yaxis.grid(True, which='major', linestyle='-', color='#CFCFCF')
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title("{}\n{}".format(filename, model_desc))
    plt.tight_layout()
    plt.savefig(filename)
    ########################################################################
    ax.clear()
    fig.clf()
    plt.close(fig)
    ########################################################################
############################################################################


############################################################################
def plot_mrg(marg_mean, marg_std, fluc_mean, fluc_std,
             ep_list, out_dir, num_units):
    ########################################################################
    model_desc = 'Margin vs Fluctuation: Units:{}'.format(num_units)
    ########################################################################
    filename = '{}/margin_fluc_U{}.png'.format(out_dir, num_units)
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ########################################################################
    marg_mean_arr = np.asarray(marg_mean)
    marg_std_arr = np.asarray(marg_std)
    fluc_mean_arr = np.asarray(fluc_mean)
    fluc_std_arr = np.asarray(fluc_std)
    ########################################################################
    m_max = (marg_mean_arr + marg_std_arr).max()
    f_max = (fluc_mean_arr + fluc_std_arr).max()
    y_max = max([m_max, f_max]) * 1.1
    ########################################################################
    m_min = (marg_mean_arr - marg_std_arr).min()
    f_min = (fluc_mean_arr - fluc_std_arr).min()
    fm_min = min([m_min, f_min])
    if (fm_min < 0.0):
        y_min = fm_min * 1.1
    else:
        y_min = fm_min * 0.9
    ########################################################################
    ax.errorbar(ep_list, marg_mean, yerr=marg_std, marker='o', markersize=3,
                alpha=0.6, label='edge margin', capthick=1, capsize=2, lw=1)
    ax.errorbar(ep_list, fluc_mean, yerr=fluc_std, marker='o', markersize=3,
                alpha=0.6, label='fluctuation', capthick=1, capsize=2, lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("distance")
    ax.set_ylim([y_min, y_max])
    ax.grid(True)
    ax.xaxis.grid(True, which='major', linestyle='-', color='#CFCFCF')
    ax.yaxis.grid(True, which='major', linestyle='-', color='#CFCFCF')
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title("{}\n{}".format(filename, model_desc))
    plt.tight_layout()
    plt.savefig(filename)
    ax.clear()
    fig.clf()
    plt.close(fig)
    ########################################################################
############################################################################


############################################################################
def split_dataset(data_ds):
    x_data = []
    t_data = []
    for lid in range(len(data_ds)):
        x_data.append(data_ds[lid][0])
        t_data.append(data_ds[lid][1])
    # end for
    x_data = np.asarray(x_data, dtype=np.float32)
    t_data = np.asarray(t_data, dtype=np.int32)
    return x_data, t_data
############################################################################


############################################################################
def main():
    ########################################################################
    reset_seed(seed_num)
    ########################################################################
    train_ds, test_ds = get_dataset(
        ds_name, model_name, data_dir, ds_size, ds_normalized)
    ########################################################################
    x_train, t_train = split_dataset(train_ds)
    x_test,  t_test  = split_dataset(test_ds)
    ########################################################################
    ## pair indexes without random label
    if ( data_type == 'train'):
        data_ds = train_ds
        t_data  = t_train
    elif (data_type == 'test'):
        data_ds = test_ds
        t_data  = t_test
    # end if
    ########################################################################
    pair_ids = make_pair_ids(trgt_num, t_data, data_dir, pair_type, data_type)
    ########################################################################
    save_files()
    ########################################################################
    data_dict = {}
    ########################################################################
    ## setup l2_init
    ########################################################################
    num_units = units_list[0]
    print('setup initial l2 values by the first network =', num_units)
    ########################################################################
    model = get_model(num_layers, num_units, num_classes, ds_size, in_ch)
    ########################################################################
    if gpu_id is not None:
        if gpu_id >= 0:
            model.to_gpu()
    ########################################################################
    l2_init, _ = calc_output(model, data_ds)
    ########################################################################
    for num_units, epoch, trig, batch in zip(
            units_list, epoch_list, trig_list, batch_list):
        ####################################################################
        num_trig = trig
        ####################################################################
        r_dict = make_data_dict()
        ####################################################################
        print('units =', num_units)
        ####################################################################
        model = get_model(num_layers, num_units, num_classes, ds_size, in_ch)
        ####################################################################
        if gpu_id is not None:
            if gpu_id >= 0:
                model.to_gpu()
        ####################################################################
        b_data = make_data_dict()
        ####################################################################
        ep_list = []
        ####################################################################
        log_name  = 'log_U{}.txt'.format(num_units)
        loss_name = 'loss_U{}.png'.format(num_units)
        acc_name  = 'accuracy_U{}.png'.format(num_units)
        ####################################################################
        train_model(train_bat, lr, epoch, model, data_ds, x_train, t_train,
                    x_test, t_test, pair_ids, out_dir, num_trig, b_data,
                    ep_list, num_units, batch, l2_init,
                    log_name, loss_name, acc_name)
        ####################################################################
        data_dict[num_units] = make_data_dict()
        data_dict[num_units]['ep_list'] = ep_list
        ####################################################################
        data_dict[num_units]['switch_mean'].append(b_data['switch_mean'])
        data_dict[num_units]['switch_std'].append(b_data['switch_std'])
        ####################################################################
        data_dict[num_units]['g_sigma_std'].append(b_data['g_sigma_std'])
        ####################################################################
        data_dict[num_units]['g_fluc_std'].append(b_data['g_fluc_std'])
        ####################################################################
        data_dict[num_units]['h_sigma_mean'].append(b_data['h_sigma_mean'])
        data_dict[num_units]['h_sigma_std'].append(b_data['h_sigma_std'])
        ####################################################################
        data_dict[num_units]['h_fluc_mean'].append(b_data['h_fluc_mean'])
        data_dict[num_units]['h_fluc_std'].append(b_data['h_fluc_std'])
        ####################################################################
        plot_err(ep_list, b_data['switch_mean'], b_data['switch_std'],
                 out_dir, 'switch', num_units)
        ####################################################################
        plot_err(ep_list, b_data['h_sigma_mean'], b_data['h_sigma_std'],
                 out_dir, 'h_sigma', num_units)
        plot_err(ep_list, b_data['h_fluc_mean'], b_data['h_fluc_std'],
                 out_dir, 'h_fluc', num_units)
        ####################################################################
        plot_data(ep_list, b_data['g_sigma_std'], out_dir,
                  'g_sigma', num_units)
        ####################################################################
        plot_diff(ep_list, b_data['g_sigma_std'],
                  b_data['h_sigma_mean'], b_data['h_sigma_std'], out_dir,
                  'gap sigma', num_units)
        plot_diff(ep_list, b_data['g_fluc_std'],
                  b_data['h_fluc_mean'], b_data['h_fluc_std'], out_dir,
                  'fluctuation', num_units)
        ####################################################################
        plot_mrg(b_data['edge_mrg_mean'], b_data['edge_mrg_std'],
                 b_data['max_fluc_mean'], b_data['max_fluc_std'],
                 ep_list, out_dir, num_units)
        ####################################################################
        fsub = '_U{}.npy'.format(num_units)
        np.save(out_dir + '/out_swt_mean'   + fsub, b_data['switch_mean'])
        np.save(out_dir + '/out_swt_std'    + fsub, b_data['switch_std'])
        np.save(out_dir + '/out_g_sig_std'  + fsub, b_data['g_sigma_std'])
        np.save(out_dir + '/out_g_flc_std'  + fsub, b_data['g_fluc_std'])
        np.save(out_dir + '/out_h_sig_mean' + fsub, b_data['h_sigma_mean'])
        np.save(out_dir + '/out_h_sig_std'  + fsub, b_data['h_sigma_std'])
        np.save(out_dir + '/out_h_flc_mean' + fsub, b_data['h_fluc_mean'])
        np.save(out_dir + '/out_h_flc_std'  + fsub, b_data['h_fluc_std'])
        np.save(out_dir + '/out_emrg_mean'  + fsub, b_data['edge_mrg_mean'])
        np.save(out_dir + '/out_emrg_std'   + fsub, b_data['edge_mrg_std'])
        np.save(out_dir + '/out_mflc_mean'  + fsub, b_data['max_fluc_mean'])
        np.save(out_dir + '/out_mflc_std'   + fsub, b_data['max_fluc_std'])
        ####################################################################
    # end for
    ########################################################################
    with open('{}/mlp_sgd_one_manual_fluctuation_units.pkl'.format(
            out_dir), 'wb') as f:
        pickle.dump(data_dict, f)
    # end with
    ########################################################################
############################################################################


############################################################################
if __name__ == '__main__':
    main()
    # import pdb; pdb.set_trace()
 
