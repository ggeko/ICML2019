# python WideResNet_fluctuation_data.py  2>&1 | tee WideResNet_fluctuation_data.log

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
import time

import json
import collections

from chainer.backends import cuda

from search_grad_1 import *
from model import *
from dataset import *

############################################################################
seed_num = 20190114
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
out_dir ='WideResNet_Fluctuation_Data_'
out_dir += datetime.datetime.today().strftime('%y%m%d%H%M%S')
os.makedirs(out_dir, exist_ok=True)
############################################################################


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
ds_amount = 50000
############################################################################
ds_normalized = True
l2_normalized = True  ## True fixed: Remark: init preserve
############################################################################


###########################################################################
## parameter for gap deviation
############################################################################
trgt_num = 128 ## gradients at begin, middle and end points
############################################################################
## parameter for divided interval
############################################################################
max_div = 1000  ## max divided number
############################################################################


############################################################################
## data set list
############################################################################
ds_list = ['MNIST', 'CIFAR10', 'CIFAR100']
############################################################################


############################################################################
# for Wide Residual networks
############################################################################
model_list = ['ResNet10', 'ResNet16', 'ResNet22',
              'ResNet28', 'ResNet34', 'ResNet40']
batch_list = [500, 500, 500, 500, 500, 500]
############################################################################
wide_factor = 1
# wide_factor = 2
############################################################################
num_calc = 50
############################################################################


############################################################################
"""
############################################################################
model_list = ['ResNet40', 'ResNet28']
batch_list = [500, 500]
num_calc = 2
############################################################################
"""
############################################################################


############################################################################
## init model parameters
############################################################################
init_rate = 0.00000001
end_rate  = 0.00000001
############################################################################
# opt = 'Adam'
opt = 'SGD'
# opt = 'MomentumSGD'
# opt = 'NesterovAG'
# opt = 'AdaDelta'
############################################################################
train_bat = 500
############################################################################
test_bat  = 500
############################################################################


############################################################################
def get_model(model_name, num_classes):
    if model_name == 'ResNet10':
        model = ResNet10(wide_factor, num_classes)
    elif model_name == 'ResNet16':
        model = ResNet16(wide_factor, num_classes)
    elif model_name == 'ResNet22':
        model = ResNet22(wide_factor, num_classes)
    elif model_name == 'ResNet28':
        model = ResNet28(wide_factor, num_classes)
    elif model_name == 'ResNet34':
        model = ResNet34(wide_factor, num_classes)
    elif model_name == 'ResNet40':
        model = ResNet40(wide_factor, num_classes)
    elif model_name == 'ResNet46':
        model = ResNet46(wide_factor, num_classes)
    elif model_name == 'ResNet52':
        model = ResNet52(wide_factor, num_classes)
    # end if
    return model
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
    in_arr = x_data.reshape((n_sample, -1))
    in_arr = xp.asarray(in_arr)
    l2_in = cpu(xp.sqrt((in_arr * in_arr).sum(axis=1)))
    rate = l2_out / l2_in
    ########################################################################
    print('input: (mean, std) =', l2_in.mean(), l2_in.std())
    print('output:(mean, std) =', l2_out.mean(), l2_out.std())
    print('rate:  (mean, std) =', rate.mean(), rate.std())
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
    print('num_acts =', num_acts)
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
def calc_fluctuation(model, pair_ids, data_ds, batch, l2_init, num_classes):
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
    hess_list, switch_list, g_sig_list = bf_new_calc(
        model, data_ds, pair_ids, l2_out, x_div, num_classes, batch)
    # hess_list, switch_list, g_sig_list, marg_list = bf_margin(
    #     model, data_ds, pair_ids, l2_out, x_div, num_classes, num_acts)
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
    result = {}
    ########################################################################
    result['switch_mean']  = switch_mean
    ########################################################################
    result['g_sigma_std']  = g_sigma_std
    result['g_fluc_std']   = g_fluc_std
    ########################################################################
    result['h_sigma_mean'] = h_sigma_mean
    result['h_fluc_mean']  = h_fluc_mean
    ########################################################################
    print('========================================================')
    g_val = 'switch:{:.2f}, g_sigma:{:.8f}, g_fluc:{:.8f}'.format(
        switch_mean, g_sigma_std, g_fluc_std)
    print(g_val)
    h_val = 'switch:{:.2f}, h_sigma:{:.8f}, h_fluc:{:.8f}'.format(
        switch_mean, h_sigma_mean, h_fluc_mean)
    print(h_val)
    print('========================================================')
    ########################################################################
    return result
############################################################################
    

############################################################################
def init_model(train_bat, model, data_ds, x_train, t_train,
               model_name, init_rate, end_rate):
    ########################################################################
    setting = '{}, bs:{}, '.format(model_name, train_bat)
    setting += 'init_rate:{}, end_rate:{}'.format(init_rate, end_rate)
    print(setting)
    ########################################################################
    optimizer = get_optimizer(opt, init_rate)
    optimizer.setup(model)
    optimizer.use_cleargrads()
    ########################################################################
    N  = x_train.shape[0]
    ########################################################################
    train_loss = []
    train_acc  = []
    ########################################################################
    print('Starting initial stabilaizer: epoch 0')
    optimizer.lr = init_rate
    delta0 = (end_rate - init_rate) / (N / train_bat)
    ########################################################################
    ########################################################################
    NN = 10000
    ########################################################################
    ########################################################################
    for eid in range(1):
        ####################################################################
        epoch_start_time = time.clock()
        ####################################################################
        perm = np.random.permutation(N)
        sum_loss = 0
        sum_acc = 0
        num = 0
        ####################################################################
        # learning loop
        ####################################################################
        for i in range(0, NN, train_bat):
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
        epoch_end_time = time.clock()
        ep_time ='time:{:.3f}'.format(epoch_end_time - epoch_start_time)
        ####################################################################
        lrate = 'lr:{:.9f}'.format(optimizer.lr)
        ####################################################################
        print_out = 'epoch:{:04d}, '.format(eid)
        print_out += (train_out + ', ' + ep_time + ', ' + lrate)
        print(print_out)
        ####################################################################
    # end for
    ########################################################################
    ## end of initialization epoch
    ########################################################################
############################################################################


############################################################################
def save_files():
    ########################################################################
    my_file = os.path.basename(__file__)
    shutil.copy('./' + my_file, './{}/'.format(out_dir) + my_file)
    cg_file = 'search_grad.py'
    shutil.copy('./' + cg_file, './{}/'.format(out_dir) + cg_file)
    md_file = 'model.py'
    shutil.copy('./' + md_file, './{}/'.format(out_dir) + md_file)
    ds_file = 'dataset.py'
    shutil.copy('./' + ds_file, './{}/'.format(out_dir) + ds_file)
    ########################################################################
############################################################################


############################################################################
def make_data_dict():
    ########################################################################
    d_dict = {}
    ########################################################################
    d_dict['switch_mean'] = []
    d_dict['switch_std']  = []
    ########################################################################
    d_dict['g_sigma_mean']  = []
    d_dict['g_sigma_std']   = []
    d_dict['g_fluc_mean']   = []
    d_dict['g_fluc_std']    = []
    ########################################################################
    d_dict['h_sigma_mean']  = []
    d_dict['h_sigma_std']   = []
    d_dict['h_fluc_mean']   = []
    d_dict['h_fluc_std']    = []
    ########################################################################
    return d_dict
############################################################################


############################################################################
def plot_diff(s_list, g_list, g_err, h_list, h_err,
              out_dir, data_name, ds_name, scale_type=''):
    ########################################################################
    if (data_name != ''):
        su_dt = data_name + '_'
        co_dt = data_name + ' '
    else:
        su_dt = ''
        co_dt = ''
    # end if
    ########################################################################
    model_desc = co_dt + 'data'
    ########################################################################
    if (scale_type == 'log'):
        filename = '{}/switch_vs_{}diff_{}_log.png'.format(
            out_dir, su_dt, ds_name)
    else:
        filename = '{}/switch_vs_{}diff_{}.png'.format(out_dir, su_dt, ds_name)
    # end if
    ########################################################################
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ########################################################################
    g_mean_arr = np.asarray(g_list)
    g_std_arr  = np.asarray(g_err)
    h_mean_arr = np.asarray(h_list)
    h_std_arr  = np.asarray(h_err)
    g_max = (g_mean_arr + g_std_arr).max()
    g_min = (g_mean_arr - g_std_arr).min()
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
    ax.errorbar(s_list, g_list, yerr=g_err, marker='o', markersize=3,
                alpha=0.6, label='grad. {}'.format(data_name),
                capthick=1, capsize=2, lw=1)
    ax.errorbar(s_list, h_list, yerr=h_err, marker='o', markersize=3,
                alpha=0.6, label='hess. {}'.format(data_name),
                capthick=1, capsize=2, lw=1)
    ax.set_xlabel('number of switches')
    ax.set_ylabel(data_name)
    ax.set_ylim([y_min, y_max])
    if (scale_type == 'log'):
        ax.set_yscale('log')
        ax.set_xscale('log')
    # end if
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
def plot_err(s_list, d_list, e_list, out_dir, data_name,
             ds_name, scale_type=''):
    ########################################################################
    if (data_name != ''):
        su_dt = data_name + '_'
        co_dt = data_name + ' '
    else:
        su_dt = ''
        co_dt = ''
    # end if
    ########################################################################
    model_desc = co_dt + 'data'
    ########################################################################
    if (scale_type == 'log'):
        filename = '{}/switch_vs_{}err_{}_log.png'.format(
            out_dir, su_dt, ds_name)
    else:
        filename = '{}/switch_vs_{}err_{}.png'.format(out_dir, su_dt, ds_name)
    # end if
    ########################################################################
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ########################################################################
    mean_arr = np.asarray(d_list)
    std_arr  = np.asarray(e_list)
    y_max = (mean_arr + std_arr).max() * 1.1
    m_min = (mean_arr - std_arr).min()
    if (m_min < 0.0):
        y_min = m_min * 1.1
    else:
        y_min = m_min * 0.9
    # end if
    ########################################################################
    ax.errorbar(s_list, d_list, yerr=e_list, marker='o', markersize=3,
                alpha=0.6, label=data_name, capthick=1, capsize=2, lw=1)
    ax.set_xlabel('number of switches')
    ax.set_ylabel(data_name)
    ax.set_ylim([y_min, y_max])
    if (scale_type == 'log'):
        ax.set_yscale('log')
        ax.set_xscale('log')
    # end if
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
def plot_data(m_list, d_dict, out_dir, data_name, err_name):
    ########################################################################
    model_desc = data_name + 'data'
    ########################################################################
    filename = '{}/dataset_diff_{}.png'.format(out_dir, data_name)
    ########################################################################
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ########################################################################
    d_max = []
    d_min = []
    for ds_name in ds_list:
        mean_arr = np.asarray(d_dict[ds_name][data_name])
        std_arr  = np.asarray(d_dict[ds_name][err_name])
        d_max.append( (mean_arr + std_arr).max() )
        d_min.append( (mean_arr - std_arr).min() )
    # end for
    y_max = max(d_max) * 1.1
    m_min = min(d_min)
    if (m_min < 0.0):
        y_min = m_min * 1.1
    else:
        y_min = m_min * 0.9
    # end if
    ########################################################################
    # m_line = list(range(len(m_list)))
    m_line = m_list
    ########################################################################
    for ds_name in ds_list:
        d_list = d_dict[ds_name][data_name]
        d_err  = d_dict[ds_name][err_name]
        ax.errorbar(m_line, d_list, yerr=d_err, marker='o', markersize=3,
                    alpha=0.6, label='{}'.format(ds_name),
                    capthick=1, capsize=2, lw=1)
    # end for
    ########################################################################
    ax.set_xlabel('model index')
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
def setup_data_ds(train_ds, test_ds, t_train, t_test):
    ########################################################################
    if ( data_type == 'train'):
        ####################################################################
        data_ds = train_ds
        t_data  = t_train
        ####################################################################
    elif (data_type == 'test'):
        ####################################################################
        data_ds = test_ds
        t_data  = t_test
        ####################################################################
    # end if
    ########################################################################
    return data_ds, t_data
############################################################################


############################################################################
def get_num_classes(ds_name):
    ########################################################################
    if (ds_name == 'MNIST'):
        ####################################################################
        data_dir = 'data_MNIST'
        num_classes = 10
        ####################################################################
    elif (ds_name == 'CIFAR10'):
        ####################################################################
        data_dir = 'data_CIFAR10'
        num_classes = 10
        ####################################################################
    elif (ds_name == 'CIFAR100'):
        ####################################################################
        data_dir = 'data_CIFAR100'
        num_classes = 100
        ####################################################################
    # end if
    ########################################################################
    os.makedirs(data_dir, exist_ok=True)
    ########################################################################
    return data_dir, num_classes
############################################################################


############################################################################
def main():
    ########################################################################
    reset_seed(seed_num)
    ########################################################################
    save_files()
    ########################################################################
    ## setup l2_init
    ########################################################################
    ds_name = 'CIFAR10'
    ########################################################################
    data_dir, num_classes = get_num_classes(ds_name)
    ########################################################################
    train_ds, test_ds = get_dataset(
        ds_name, 'ResNet', data_dir, ds_size, ds_normalized, ds_amount)
    ########################################################################
    x_train, t_train = split_dataset(train_ds)
    x_test,  t_test  = split_dataset(test_ds)
    ########################################################################
    data_ds, t_data = setup_data_ds(train_ds, test_ds, t_train, t_test)
    ########################################################################
    model_name = 'ResNet10'
    print('setup initial l2 values by the network :', model_name)
    ########################################################################
    model = get_model(model_name, num_classes)
    ########################################################################
    if gpu_id is not None:
        if gpu_id >= 0:
            model.to_gpu()
    ########################################################################
    init_model(train_bat, model, data_ds, x_train, t_train,
               model_name, init_rate, end_rate)
    ########################################################################
    l2_init, _ = calc_output(model, data_ds)
    ########################################################################

    ########################################################################
    ## start calculation
    ########################################################################
    d_dict = {}
    ########################################################################
    for ds_name in ds_list:
        ####################################################################
        d_dict[ds_name] = make_data_dict()
        ####################################################################
        data_dir, num_classes = get_num_classes(ds_name)
        ####################################################################
        train_ds, test_ds = get_dataset(
            ds_name, 'ResNet', data_dir, ds_size, ds_normalized, ds_amount)
        ####################################################################
        x_train, t_train = split_dataset(train_ds)
        x_test,  t_test  = split_dataset(test_ds)
        ####################################################################
        data_ds, t_data = setup_data_ds(train_ds, test_ds, t_train, t_test)
        ####################################################################
        pair_ids = make_pair_ids(trgt_num, t_data, data_dir,
                                 pair_type, data_type, ds_amount)
        ####################################################################
        for model_name, batch in zip(model_list, batch_list):
            ################################################################
            swt_arr = []
            ################################################################
            g_sig_arr = []
            g_flc_arr = []
            ################################################################
            h_sig_arr = []
            h_flc_arr = []
            ################################################################
            for cid in range(num_calc):
                ############################################################
                start_time = time.clock()
                ############################################################
                print('ds_name:{}, model_name :{}, No.{}'.format(
                    ds_name, model_name, cid))
                ############################################################
                model = get_model(model_name, num_classes)
                ############################################################
                if gpu_id is not None:
                    if gpu_id >= 0:
                        model.to_gpu()
                ############################################################
                init_model(train_bat, model, data_ds, x_train, t_train,
                   model_name, init_rate, end_rate)
                ############################################################
                result = calc_fluctuation(
                    model, pair_ids, data_ds, batch, l2_init, num_classes)
                ############################################################
                swt_arr.append(result['switch_mean'])
                ############################################################
                g_sig_arr.append(result['g_sigma_std'])
                g_flc_arr.append(result['g_fluc_std'])
                ############################################################
                h_sig_arr.append(result['h_sigma_mean'])
                h_flc_arr.append(result['h_fluc_mean'])
                ############################################################
                end_time = time.clock()
                print('time:{:.3f}\n'.format(end_time - start_time))
                ############################################################
            # end for
            ################################################################
            swt_arr = np.asarray(swt_arr)
            ################################################################
            g_sig_arr = np.asarray(g_sig_arr)
            g_flc_arr = np.asarray(g_flc_arr)
            ################################################################
            h_sig_arr = np.asarray(h_sig_arr)
            h_flc_arr = np.asarray(h_flc_arr)
            ################################################################
            d_dict[ds_name]['switch_mean'].append(swt_arr.mean())
            d_dict[ds_name]['switch_std'].append(swt_arr.std())
            ################################################################
            d_dict[ds_name]['g_sigma_mean'].append(g_sig_arr.mean())
            d_dict[ds_name]['g_sigma_std'].append(g_sig_arr.std())
            d_dict[ds_name]['g_fluc_mean'].append(g_flc_arr.mean())
            d_dict[ds_name]['g_fluc_std'].append(g_flc_arr.std())
            ################################################################
            d_dict[ds_name]['h_sigma_mean'].append(h_sig_arr.mean())
            d_dict[ds_name]['h_sigma_std'].append(h_sig_arr.std())
            d_dict[ds_name]['h_fluc_mean'].append(h_flc_arr.mean())
            d_dict[ds_name]['h_fluc_std'].append(h_flc_arr.std())
            ################################################################
            out_base = '_{}_{}.npy'.format(model_name, ds_name)
            np.save(out_dir + '/out_swt'   + out_base, swt_arr)
            np.save(out_dir + '/out_g_sig' + out_base, g_sig_arr)
            np.save(out_dir + '/out_g_flc' + out_base, g_flc_arr)
            np.save(out_dir + '/out_h_sig' + out_base, h_sig_arr)
            np.save(out_dir + '/out_h_flc' + out_base, h_flc_arr)
            ################################################################
        # end for
        ####################################################################
        plot_err(d_dict[ds_name]['switch_mean'],
                 d_dict[ds_name]['g_sigma_mean'],
                 d_dict[ds_name]['g_sigma_std'],
                 out_dir, 'g_sigma', ds_name, 'log')
        ####################################################################
        plot_err(d_dict[ds_name]['switch_mean'],
                 d_dict[ds_name]['g_sigma_mean'],
                 d_dict[ds_name]['g_sigma_std'],
                 out_dir, 'g_sigma', ds_name)
        plot_err(d_dict[ds_name]['switch_mean'],
                 d_dict[ds_name]['g_fluc_mean'],
                 d_dict[ds_name]['g_fluc_std'],
                 out_dir, 'g_fluc', ds_name)
        ####################################################################
        plot_err(d_dict[ds_name]['switch_mean'],
                 d_dict[ds_name]['h_sigma_mean'],
                 d_dict[ds_name]['h_sigma_std'],
                 out_dir, 'h_sigma', ds_name, 'log')
        ####################################################################
        plot_err(d_dict[ds_name]['switch_mean'],
                 d_dict[ds_name]['h_sigma_mean'],
                 d_dict[ds_name]['h_sigma_std'],
                 out_dir, 'h_sigma', ds_name)
        plot_err(d_dict[ds_name]['switch_mean'],
                 d_dict[ds_name]['h_fluc_mean'],
                 d_dict[ds_name]['h_fluc_std'],
                 out_dir, 'h_fluc', ds_name)
        ####################################################################
        plot_diff(d_dict[ds_name]['switch_mean'],
                  d_dict[ds_name]['g_sigma_mean'],
                  d_dict[ds_name]['g_sigma_std'],
                  d_dict[ds_name]['h_sigma_mean'],
                  d_dict[ds_name]['h_sigma_std'],
                  out_dir, 'gap sigma', ds_name, 'log')
        ####################################################################
        plot_diff(d_dict[ds_name]['switch_mean'],
                  d_dict[ds_name]['g_sigma_mean'],
                  d_dict[ds_name]['g_sigma_std'],
                  d_dict[ds_name]['h_sigma_mean'],
                  d_dict[ds_name]['h_sigma_std'],
                  out_dir, 'gap sigma', ds_name)
        plot_diff(d_dict[ds_name]['switch_mean'],
                  d_dict[ds_name]['g_fluc_mean'],
                  d_dict[ds_name]['g_fluc_std'],
                  d_dict[ds_name]['h_fluc_mean'],
                  d_dict[ds_name]['h_fluc_std'],
                  out_dir, 'fluctuation', ds_name)
        ####################################################################
        file_name = '{}/wide_res_net_fluctuation_data_{}.pkl'.format(
            out_dir, ds_name)
        with open(file_name, 'wb') as f:
            pickle.dump(d_dict[ds_name], f)
        # end with
        ####################################################################
    # end for
    ########################################################################
    file_name = '{}/wide_res_net_fluctuation_data_all.pkl'.format(out_dir)
    with open(file_name, 'wb') as f:
        pickle.dump(d_dict, f)
    # end with
    ########################################################################
    plot_data(model_list, d_dict, out_dir, 'switch_mean',  'switch_std')
    plot_data(model_list, d_dict, out_dir, 'g_sigma_mean', 'g_sigma_std')
    plot_data(model_list, d_dict, out_dir, 'h_sigma_mean', 'h_sigma_std')
    plot_data(model_list, d_dict, out_dir, 'g_fluc_mean',  'g_fluc_std')
    plot_data(model_list, d_dict, out_dir, 'h_fluc_mean',  'h_fluc_std')
    ########################################################################
############################################################################


############################################################################
############################################################################
if __name__ == '__main__':
    main()
    # import pdb; pdb.set_trace()
