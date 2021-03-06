# python MLP_fluctuation_data.py 2>&1 | tee MLP_fluctuation_data.log2
import numpy as np
import matplotlib.pyplot as plt

import chainer
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.dataset import concat_examples

import os
import random
import copy
import datetime
import code
import sys
import shutil
import pickle
import time

from chainer.backends import cuda

from search_grad import *
from model import *
from dataset import *


############################################################################
seed_num = 20190114
############################################################################


############################################################################
gpu_id = 0
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
out_dir ='MLP_Fluctuation_Data_'
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
l2_normalized = True ## Remark: fix to True
############################################################################


###########################################################################
## parameter for gap deviation
############################################################################
trgt_num = 128 ## gradients at begin, middle and end points
############################################################################
## parameter for divided interval
############################################################################
max_div  = 1000  ## max divided number
############################################################################


############################################################################
## data set list
############################################################################
ds_list = ['MNIST', 'CIFAR10', 'CIFAR100']
############################################################################


############################################################################
# MLP networks
############################################################################
model_name = 'MLP'
# num_units = 1000
num_layers = 2
############################################################################
units_list = [1000, 2000, 3000, 4000, 5000]
############################################################################
batch_list = [ 500,  500,  500,  500,  500]
############################################################################
num_calc = 100
############################################################################


############################################################################
"""
############################################################################
units_list = [3000, 7000, 5000]
batch_list = [ 500,  500,  500]
num_calc = 5
############################################################################
"""
############################################################################


############################################################################
def get_model(num_layers, num_units, num_classes):
    return MLP(num_layers, num_units, num_classes)
############################################################################


############################################################################
def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    cuda.cupy.random.seed(seed)
############################################################################


############################################################################
def calc_output(model, train_ds):
    ########################################################################
    x_train = []
    for lid in range(len(train_ds)):
        x_train.append(train_ds[lid][0])
    # end for
    x_train = np.asarray(x_train, dtype=np.float32)
    ########################################################################
    # NN's value
    nn = lambda x: model(x)
    ########################################################################
    ## get output values
    output_val = []
    n_sample = x_train.shape[0]
    for i in range(0, n_sample, test_bat):
        ####################################################################
        xx = Variable(xp.asarray(x_train[i:i + test_bat].astype(np.float32)))
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
    in_arr = x_train.reshape((n_sample, -1))
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
    xx = Variable(xp.asarray(x_train[0:1].astype(np.float32)))
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
    # hess_list, switch_list, g_sig_list = bf_new_calc(
    #    model, data_ds, pair_ids, l2_out, x_div, num_classes, batch)
    hess_list, switch_list, g_sig_list, marg_list = bf_margin(
        model, data_ds, pair_ids, l2_out, x_div, num_classes, num_acts)
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
def plot_diff(u_list, g_list, g_err, h_list, h_err,
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
        filename = '{}/units_{}diff_{}_log.png'.format(out_dir, su_dt, ds_name)
    else:
        filename = '{}/units_{}diff_{}.png'.format(out_dir, su_dt, ds_name)
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
    ax.errorbar(u_list, g_list, yerr=g_err, marker='o', markersize=3,
                alpha=0.6, label='grad. {}'.format(data_name),
                capthick=1, capsize=2, lw=1)
    ax.errorbar(u_list, h_list, yerr=h_err, marker='o', markersize=3,
                alpha=0.6, label='hess. {}'.format(data_name),
                capthick=1, capsize=2, lw=1)
    ax.set_xlabel('number of units')
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
def plot_err(u_list, d_list, e_list, out_dir, data_name,
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
        filename = '{}/{}data_{}_log.png'.format(out_dir, su_dt, ds_name)
    else:
        filename = '{}/{}data_{}.png'.format(out_dir, su_dt, ds_name)
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
    ax.errorbar(u_list, d_list, yerr=e_list, marker='o', markersize=3,
                alpha=0.6, label=data_name, capthick=1, capsize=2, lw=1)
    ax.set_xlabel('number of units')
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
def plot_data(u_list, d_dict, out_dir, data_name, err_name):
    ########################################################################
    model_desc = data_name + ' data'
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
    for ds_name in ds_list:
        d_list = d_dict[ds_name][data_name]
        d_err  = d_dict[ds_name][err_name]
        ax.errorbar(u_list, d_list, yerr=d_err, marker='o', markersize=3,
                    alpha=0.6, label='{}'.format(ds_name),
                    capthick=1, capsize=2, lw=1)
    # end for
    ########################################################################
    ax.set_xlabel('number of units')
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
        ds_name, model_name, data_dir, ds_size, ds_normalized, ds_amount)
    ########################################################################
    x_train, t_train = split_dataset(train_ds)
    x_test,  t_test  = split_dataset(test_ds)
    ########################################################################
    data_ds, t_data = setup_data_ds(train_ds, test_ds, t_train, t_test)
    ########################################################################
    num_units = units_list[0]
    ########################################################################
    model = get_model(num_layers, num_units, num_classes)
    ########################################################################
    if gpu_id is not None:
        if gpu_id >= 0:
            model.to_gpu()
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
            ds_name, model_name, data_dir, ds_size, ds_normalized, ds_amount)
        ####################################################################
        x_train, t_train = split_dataset(train_ds)
        x_test,  t_test  = split_dataset(test_ds)
        ####################################################################
        data_ds, t_data = setup_data_ds(train_ds, test_ds, t_train, t_test)
        ####################################################################
        pair_ids = make_pair_ids(trgt_num, t_data, data_dir,
                                 pair_type, data_type, ds_amount)
        ####################################################################
        for num_units, batch in zip(units_list, batch_list):
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
                print('ds_name: {}, units: {}, No.{}'.format(
                    ds_name, num_units, cid))
                ############################################################
                model = get_model(num_layers, num_units, num_classes)
                ############################################################
                if gpu_id is not None:
                    if gpu_id >= 0:
                        model.to_gpu()
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
            out_base = '_U{}_{}.npy'.format(num_units, ds_name)
            np.save(out_dir + '/out_swt'   + out_base, swt_arr)
            np.save(out_dir + '/out_g_sig' + out_base, g_sig_arr)
            np.save(out_dir + '/out_g_flc' + out_base, g_flc_arr)
            np.save(out_dir + '/out_h_sig' + out_base, h_sig_arr)
            np.save(out_dir + '/out_h_flc' + out_base, h_flc_arr)
            ################################################################
        # end for
        ####################################################################
        plot_err(units_list, d_dict[ds_name]['switch_mean'],
                 d_dict[ds_name]['switch_std'],
                 out_dir, 'switch', ds_name)
        ####################################################################
        plot_err(units_list, d_dict[ds_name]['g_sigma_mean'],
                 d_dict[ds_name]['g_sigma_std'],
                 out_dir, 'g_sigma', ds_name)
        plot_err(units_list, d_dict[ds_name]['g_fluc_mean'],
                 d_dict[ds_name]['g_fluc_std'],
                 out_dir, 'g_fluc', ds_name)
        ####################################################################
        plot_err(units_list, d_dict[ds_name]['h_sigma_mean'],
                 d_dict[ds_name]['h_sigma_std'],
                 out_dir, 'h_sigma', ds_name)
        plot_err(units_list, d_dict[ds_name]['h_fluc_mean'],
                 d_dict[ds_name]['h_fluc_std'],
                 out_dir, 'h_fluc', ds_name)
        ####################################################################
        plot_diff(units_list, d_dict[ds_name]['g_sigma_mean'],
                  d_dict[ds_name]['g_sigma_std'],
                  d_dict[ds_name]['h_sigma_mean'],
                  d_dict[ds_name]['h_sigma_std'],
                  out_dir, 'gap sigma', ds_name)
        ####################################################################
        plot_diff(units_list, d_dict[ds_name]['g_fluc_mean'],
                  d_dict[ds_name]['g_fluc_std'],
                  d_dict[ds_name]['h_fluc_mean'],
                  d_dict[ds_name]['h_fluc_std'],
                  out_dir, 'fluctuation', ds_name)
        ####################################################################
        file_name = '{}/mlp_fluctuation_data_{}.pkl'.format(out_dir, ds_name)
        with open(file_name, 'wb') as f:
            pickle.dump(d_dict[ds_name], f)
        # end with
        ####################################################################
    # end for
    ########################################################################
    plot_data(units_list, d_dict, out_dir, 'switch_mean',  'switch_std')
    plot_data(units_list, d_dict, out_dir, 'g_sigma_mean', 'g_sigma_std')
    plot_data(units_list, d_dict, out_dir, 'h_sigma_mean', 'h_sigma_std')
    plot_data(units_list, d_dict, out_dir, 'g_fluc_mean',  'g_fluc_std')
    plot_data(units_list, d_dict, out_dir, 'h_fluc_mean',  'h_fluc_std')
    ########################################################################
    file_name = '{}/mlp_fluctuation_data_all.pkl'.format(out_dir)
    with open(file_name, 'wb') as f:
        pickle.dump(d_dict, f)
    # end with
    ########################################################################
############################################################################    

    
############################################################################
############################################################################
if __name__ == '__main__':
    main()
    # import pdb; pdb.set_trace()
 
