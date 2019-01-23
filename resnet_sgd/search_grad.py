#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import sys
import numpy as np
import chainer
from chainer import Variable
from chainer.backends import cuda

xp = cuda.cupy
cpu = cuda.to_cpu


#############################################################################
gpu_id = 0
#############################################################################
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

test_bat = 1024

############################################################################
############################################################################
def get_nodes_data(model, train_ds, pair_ids):
    ########################################################################
    ## for fluctuation data
    ########################################################################
    beg_val = []
    end_val = []
    for j in range(pair_ids.shape[0]):
        t_idx = pair_ids[j][0]
        n_idx = pair_ids[j][1]
        t_img = train_ds[t_idx][0]
        n_img = train_ds[n_idx][0]
        beg_val.append(t_img)
        end_val.append(n_img)
    # end for
    beg_arr = np.asarray(beg_val, dtype=np.float32)
    end_arr = np.asarray(end_val, dtype=np.float32)
    n_sample = beg_arr.shape[0]
    # NN's value
    nn = lambda x: model.get_act(x)
    ########################################################################
    ## get output and  ReLU on off
    ########################################################################
    beg_act = []
    end_act = []
    dif_act = []
    ########################################################################
    for i in range(0, n_sample, test_bat):
        ####################################################################
        bb = Variable(xp.asarray(beg_arr[i:i + test_bat].astype(np.float32)))
        ee = Variable(xp.asarray(end_arr[i:i + test_bat].astype(np.float32)))
        ####################################################################
        with chainer.using_config('train', False):
            ################################################################
            beg_ro = nn(bb)
            ################################################################
            ro_arr = [val.reshape(val.shape[0], -1) for val in beg_ro]
            beg_act.append(cpu(xp.concatenate(ro_arr, axis=1)))
            ################################################################
            end_ro = nn(ee)
            ################################################################
            ro_arr = [val.reshape(val.shape[0], -1) for val in end_ro]
            end_act.append(cpu(xp.concatenate(ro_arr, axis=1)))
            ################################################################
        # end with
        ####################################################################
    # end for
    ########################################################################
    beg_act = np.concatenate(beg_act, axis=0)
    end_act = np.concatenate(end_act, axis=0)
    ########################################################################
    relu_sw = np.sum((beg_act != end_act), axis=1)
    same_sw = np.sum((beg_act == end_act), axis=1)
    ########################################################################
    relu_sw = np.asarray(relu_sw, dtype=np.float)
    print('relu_sw :', relu_sw.shape)
    ########################################################################
    return relu_sw, same_sw
#############################################################################
############################################################################
def get_points_data(model, train_ds, pair_ids, l2_out, margin_num):
    ########################################################################
    ## for margin data
    ########################################################################
    nn = lambda x: model(x)
    mrg_out = []
    l2_val = []
    for j in range(pair_ids.shape[0]):
        t_idx = pair_ids[j][0]
        n_idx = pair_ids[j][1]
        t_img = train_ds[t_idx][0]
        n_img = train_ds[n_idx][0]
        mrg_val = []
        l2_bit = []
        for m in range(margin_num):
            t = m / margin_num
            mrg_val.append(t_img * (1-t) + n_img * t)
            l2_bit.append(l2_out[t_idx] * (1-t) +  l2_out[n_idx] * t)
        # end for
        mrg_arr = np.asarray(mrg_val, dtype=np.float32)
        l2_bit = np.asarray(l2_bit)
        ####################################################################
        ## get output
        mrg = Variable(xp.asarray(mrg_arr))
        ####################################################################
        with chainer.using_config('train', False):
            mrg_out.append(cpu(nn(mrg).data))
        # end with
        l2_val.append(l2_bit)
        ####################################################################
    # end for
    ########################################################################
    ## normalized by output values
    l2_val = np.asarray(l2_val).reshape((pair_ids.shape[0], margin_num, 1))
    margin_arr = np.asarray(mrg_out, dtype=np.float32) / l2_val
    ########################################################################
    ## for fluctuation data
    ########################################################################
    beg_val = []
    end_val = []
    mid_val = []
    beg_l2 = []
    end_l2 = []
    mid_l2 = []
    for j in range(pair_ids.shape[0]):
        t_idx = pair_ids[j][0]
        n_idx = pair_ids[j][1]
        t_img = train_ds[t_idx][0]
        n_img = train_ds[n_idx][0]
        beg_val.append(t_img)
        end_val.append(n_img)
        mid_val.append((t_img + n_img) / 2.0)
        beg_l2.append(l2_out[t_idx])
        end_l2.append(l2_out[n_idx])
        mid_l2.append((l2_out[t_idx] + l2_out[n_idx]) / 2.0)
    # end for
    beg_arr = np.asarray(beg_val, dtype=np.float32)
    end_arr = np.asarray(end_val, dtype=np.float32)
    mid_arr = np.asarray(mid_val, dtype=np.float32)
    beg_l2 = np.asarray(beg_l2, dtype=np.float32).reshape((-1,1))
    end_l2 = np.asarray(end_l2, dtype=np.float32).reshape((-1,1))
    mid_l2 = np.asarray(mid_l2, dtype=np.float32).reshape((-1,1))
    n_sample = beg_arr.shape[0]
    # NN's value
    nn = lambda x: model.get_act(x)
    ########################################################################
    ## get output and  ReLU on off
    beg_out = []
    end_out = []
    ########################################################################
    beg_cut = {}
    end_cut = {}
    mid_cut = {}
    ########################################################################
    beg_act = []
    end_act = []
    ########################################################################
    for i in range(0, n_sample, test_bat):
        ####################################################################
        bb = Variable(xp.asarray(beg_arr[i:i + test_bat].astype(np.float32)))
        ee = Variable(xp.asarray(end_arr[i:i + test_bat].astype(np.float32)))
        mm = Variable(xp.asarray(mid_arr[i:i + test_bat].astype(np.float32)))
        ####################################################################
        with chainer.using_config('train', False):
            ################################################################
            beg_ro = nn(bb)
            beg_out.append(model.output)
            ################################################################
        # end with
        ####################################################################
        beg_cut[i] = beg_ro
        ####################################################################
        ro_arr = [val.reshape(val.shape[0], -1) for val in beg_ro]
        beg_act.append(cpu(xp.concatenate(ro_arr, axis=1)))
        ####################################################################
        with chainer.using_config('train', False):
            ################################################################
            end_ro = nn(ee)
            end_out.append(model.output)
            ################################################################
        # end with
        ####################################################################
        end_cut[i] = end_ro
        ####################################################################
        ro_arr = [val.reshape(val.shape[0], -1) for val in end_ro]
        end_act.append(cpu(xp.concatenate(ro_arr, axis=1)))
        ####################################################################
        with chainer.using_config('train', False):
            ################################################################
            mid_cut[i] = nn(mm)
            ################################################################
        # end with
        ####################################################################
    # end for
    ########################################################################
    beg_act = np.concatenate(beg_act, axis=0)
    end_act = np.concatenate(end_act, axis=0)
    same_sw = np.sum((beg_act == end_act), axis=1)
    relu_sw = np.sum((beg_act != end_act), axis=1)
    ########################################################################
    ## normalized by output values
    beg_nn = cpu(xp.concatenate(beg_out, axis=0)) / beg_l2
    end_nn = cpu(xp.concatenate(end_out, axis=0)) / end_l2
    ########################################################################
    # get extend network value by cut 
    # NN's value
    nn = lambda x, cut : model.cutoff(x, cut)
    # get nn extended activation
    beg_ll = []
    end_rr = []
    mid_rr = []
    mid_ll = []
    ########################################################################
    for i in range(0, n_sample, test_bat):
        ####################################################################
        rr = Variable(xp.asarray(beg_arr[i:i + test_bat].astype(np.float32)))
        ll = Variable(xp.asarray(end_arr[i:i + test_bat].astype(np.float32)))
        ####################################################################
        b_cut_arr = beg_cut[i]
        e_cut_arr = end_cut[i]
        m_cut_arr = mid_cut[i]
        ####################################################################
        with chainer.using_config('train', False):
            ################################################################
            beg_ll.append(cpu(nn(ll, b_cut_arr).data))
            end_rr.append(cpu(nn(rr, e_cut_arr).data))
            mid_rr.append(cpu(nn(rr, m_cut_arr).data))
            mid_ll.append(cpu(nn(ll, m_cut_arr).data))
            ################################################################
        # end with
        ####################################################################
    # end for
    ########################################################################
    ## normalized by output values
    beg_ll = np.concatenate(beg_ll, axis=0).astype(np.float) / beg_l2
    end_rr = np.concatenate(end_rr, axis=0).astype(np.float) / end_l2
    mid_rr = np.concatenate(mid_rr, axis=0).astype(np.float) / mid_l2
    mid_ll = np.concatenate(mid_ll, axis=0).astype(np.float) / mid_l2
    ########################################################################
    ## calc gradients
    beg_grad = []
    end_grad = []
    mid_grad = []
    df_list = []
    ########################################################################
    for j in range(pair_ids.shape[0]):
        ####################################################################
        t_idx = pair_ids[j][0]
        n_idx = pair_ids[j][1]
        t_img = train_ds[t_idx][0]
        n_img = train_ds[n_idx][0]
        ####################################################################
        rl_diff = (t_img - n_img).reshape((-1,))
        ####################################################################
        df_val = np.linalg.norm(rl_diff)
        ####################################################################
        b_r = beg_nn[j]
        b_l = beg_ll[j]
        e_r = end_rr[j]
        e_l = end_nn[j]
        m_r = mid_rr[j]
        m_l = mid_ll[j]
        beg_grad.append( (b_l - b_r) / df_val )
        end_grad.append( (e_l - e_r) / df_val )
        mid_grad.append( (m_l - m_r) / df_val )
        df_list.append(df_val)
    # end for
    ########################################################################
    beg_grd = np.asarray(beg_grad, dtype=np.float)
    end_grd = np.asarray(end_grad, dtype=np.float)
    mid_grd = np.asarray(mid_grad, dtype=np.float)
    relu_sw = np.asarray(relu_sw, dtype=np.float)
    df_arr = np.asarray(df_list, dtype=np.float)
    ########################################################################
    return beg_grd, end_grd, mid_grd, relu_sw, df_arr, margin_arr
#############################################################################


#############################################################################
def net_behavior(model, img_a, img_b, x_div):
    #########################################################################
    test_bat = 20
    #########################################################################
    ## data setup
    #########################################################################
    t_img = chainer.links.model.vision.resnet.prepare(img_a)
    n_img = chainer.links.model.vision.resnet.prepare(img_b)
    #########################################################################
    rl_diff = (t_img - n_img).reshape((-1,))
    #########################################################################
    df_val = np.linalg.norm(rl_diff)
    #########################################################################
    num_sample = x_div.shape[0]
    #########################################################################
    
    ########################################################################
    ## check ReLU on off and calc extended gradient data
    ########################################################################
    # NN's value
    nn_f = lambda x: model.get_act(x)
    nn_c = lambda x, cut: model.cutoff(x, cut)
    ########################################################################
    data_arr = []
    grad_arr = []
    ########################################################################
    all_index = np.arange(num_sample, dtype=np.int)
    ########################################################################
    print('calc. gradient {}:'.format(num_sample), end='', flush=True)
    ########################################################################
    for i in range(0, num_sample, test_bat):
        ####################################################################
        print('{},'.format(i), end='', flush=True)
        ####################################################################
        sub_idx = all_index[i:i + test_bat]
        ####################################################################
        sub_div = x_div[sub_idx]
        x_val = np.array([(1 - t) * t_img + t * n_img for t in sub_div])
        xf = chainer.Variable(xp.asarray(x_val.astype(np.float32)))
        ####################################################################
        x_drc = [ n_img - t_img ] * sub_idx.shape[0]
        xc = chainer.Variable(xp.asarray(x_drc))
        ####################################################################
        with chainer.using_config('train', False):
            ################################################################
            ro_data = nn_f(xf)
            data_arr.append(model.output)
            ################################################################
            grad_arr.append( nn_c(xc, ro_data).data )
            ################################################################
        # end with
        ####################################################################
    # end for
    ########################################################################
    print(' :end')
    ########################################################################
    d_shape = (x_div.shape[0], -1)
    ########################################################################
    data_arr = xp.concatenate(data_arr, axis=0).reshape(d_shape)
    data_arr = cpu(data_arr)
    ########################################################################
    grad_arr = xp.concatenate(grad_arr, axis=0).reshape(d_shape)
    grad_arr = cpu(grad_arr) / df_val
    ########################################################################
    return data_arr, grad_arr
############################################################################


#############################################################################
def bf_behavior(model, img_a, img_b, x_div, num_out, test_bat):
    #########################################################################
    ## data setup
    #########################################################################
    t_img = chainer.links.model.vision.resnet.prepare(img_a)
    n_img = chainer.links.model.vision.resnet.prepare(img_b)
    #########################################################################
    rl_diff = (t_img - n_img).reshape((-1,))
    #########################################################################
    df_val = np.linalg.norm(rl_diff)
    #########################################################################
    num_sample = x_div.shape[0]
    #########################################################################
    
    ########################################################################
    ## check ReLU on off and calc extended gradient data
    ########################################################################
    # NN's value
    nn_f = lambda x: model.get_act(x)
    nn_c = lambda x, cut: model.cutoff(x, cut)
    ########################################################################
    data_arr = []
    grad_arr = []
    ########################################################################
    node_list = []
    hess_list = []
    ########################################################################
    last_bool = None
    last_grad = None
    ########################################################################
    all_index = np.arange(num_sample, dtype=np.int)
    ########################################################################
    print('calc. gradient {}:'.format(num_sample), end='', flush=True)
    ########################################################################
    for i in range(0, num_sample, test_bat):
        ####################################################################
        print('{},'.format(i), end='', flush=True)
        ####################################################################
        sub_idx = all_index[i:i + test_bat]
        ####################################################################
        sub_div = x_div[sub_idx]
        x_val = np.array([(1 - t) * t_img + t * n_img for t in sub_div])
        xf = chainer.Variable(xp.asarray(x_val.astype(np.float32)))
        ####################################################################
        x_drc = [ n_img - t_img ] * sub_idx.shape[0]
        xc = chainer.Variable(xp.asarray(x_drc))
        ####################################################################
        with chainer.using_config('train', False):
            ################################################################
            ro_data = nn_f(xf)
            data_arr.append(model.output)
            ################################################################
            grad_arr.append( nn_c(xc, ro_data).data )
            grad_data = grad_arr[-1]
            ################################################################
        # end with
        ####################################################################
        ro_arr = [val.reshape(val.shape[0], -1) for val in ro_data]
        x_bool = xp.concatenate(ro_arr, axis=1)
        ####################################################################
        if (last_bool is not None):
            x_bool = xp.concatenate([last_bool, x_bool], axis=0)
            grad_data = xp.concatenate([last_grad, grad_data], axis=0)
        # end if
        ####################################################################
        bool0 = x_bool[:-1]
        bool1 = x_bool[1:]
        trans_arr = xp.sum((bool0 != bool1), axis=1)
        ####################################################################
        for did, n_trans in enumerate(trans_arr):
            ################################################################
            if (n_trans > 0):
                ############################################################
                node_list.append(cpu(n_trans))
                hess = (grad_data[did+1] - grad_data[did]) / df_val
                hess_list.append(cpu(hess))
                ############################################################
            # end if
            ################################################################
        # end for
        ####################################################################
        last_bool = x_bool[-1:]
        last_grad = grad_data[-1:]
        ####################################################################
    # end for
    ########################################################################
    print(' :end')
    ########################################################################
    d_shape = (x_div.shape[0], -1)
    ########################################################################
    data_arr = xp.concatenate(data_arr, axis=0).reshape(d_shape)
    data_arr = cpu(data_arr)
    ########################################################################
    grad_arr = xp.concatenate(grad_arr, axis=0).reshape(d_shape)
    grad_arr = cpu(grad_arr) / df_val
    ########################################################################
    
    ########################################################################
    hess_arr = np.array(hess_list).reshape((-1, num_out))
    node_arr = np.array(node_list).reshape((-1, 1))
    ########################################################################
    switch = node_arr.sum()
    hess_sq = (hess_arr * hess_arr) / node_arr
    hess_sig = np.sqrt(hess_sq.sum(axis=0) / len(node_arr))
    ########################################################################
    
    ########################################################################
    beg_grd = grad_arr[0]
    end_grd = grad_arr[-1]
    ########################################################################
    ## for mid gradient
    ########################################################################
    mid_switch = switch / 2.0
    sum_switch = 0
    mid_id = None
    mid_mean = False
    ########################################################################
    for did, n_trans in enumerate(node_list):
        ####################################################################
        pre_val = mid_switch - sum_switch 
        sum_switch += n_trans
        ####################################################################
        if (sum_switch > mid_switch):
            post_val = sum_switch - mid_switch
            if (pre_val < post_val):
                mid_id = did - 1
            elif (pre_val > post_val):
                mid_id = did
            elif (pre_val == post_val):
                mid_id = did
                mid_mean = True
            # end if
            break
        # end if
        ####################################################################
    # end for
    ########################################################################
    if (mid_mean == False):
        mid_grd = grad_arr[mid_id]
    else:
        mid_grd = (grad_arr[mid_id - 1] + grad_arr[mid_id]) / 2
    # end if
    ########################################################################
    linear_mid = (beg_grd + end_grd) / 2
    g_diverge = mid_grd - linear_mid
    g_sigma = g_diverge / (np.sqrt(switch) / 2)
    ########################################################################
    hess_list = hess_sig.tolist()
    g_sig_list = g_sigma.tolist()
    ########################################################################
    return data_arr, grad_arr, hess_list, switch, g_sig_list
############################################################################


#############################################################################
def bf_new_calc(model, train_ds, pair_ids, l2_out, x_div, num_classes, batch):
    #########################################################################
    switch_data = []
    hess_data   = []
    g_sig_data  = []
    ########################################################################
    num_sample = x_div.shape[0]
    all_index = np.arange(num_sample, dtype=np.int)
    ########################################################################
    print('calc. gradient {}:'.format(
        pair_ids.shape[0]), end='', flush=True)
    ########################################################################
    # NN's value
    nn_f = lambda x: model.get_act(x)
    nn_c = lambda x, cut: model.cutoff(x, cut)
    ########################################################################
    for j in range(pair_ids.shape[0]):
        ####################################################################
        ## data setup
        ####################################################################
        t_idx = pair_ids[j][0]
        n_idx = pair_ids[j][1]
        t_img = train_ds[t_idx][0]
        n_img = train_ds[n_idx][0]
        ####################################################################
        grad_arr  = []
        l2_arr    = []
        node_list = []
        hess_list = []
        ####################################################################
        last_bool = None
        last_grad = None
        last_l2   = None
        ####################################################################
        rl_diff = (t_img - n_img).reshape((-1,))
        ####################################################################
        df_val = np.linalg.norm(rl_diff)
        ####################################################################
        ## check ReLU on off and calc extended gradient data
        ####################################################################
        if (j%10 == 0):
            print('{},'.format(j), end='', flush=True)
        # end if
        ####################################################################
        for i in range(0, num_sample, batch):
            ################################################################
            sub_idx = all_index[i:i + batch]
            ################################################################
            sub_div = x_div[sub_idx]
            x_val = np.array([(1 - t) * t_img + t * n_img for t in sub_div])
            xf = chainer.Variable(xp.asarray(x_val.astype(np.float32)))
            ################################################################
            l2_val =  [(1 - t) * l2_out[t_idx] + t * l2_out[n_idx]
                       for t in sub_div]
            l2_arr.extend( l2_val )
            l2_val = xp.reshape(xp.asarray(l2_val), (-1, 1))
            ################################################################
            x_drc = [ n_img - t_img ] * sub_idx.shape[0]
            xc = chainer.Variable(xp.asarray(x_drc))
            ################################################################
            with chainer.using_config('train', False):
                ############################################################
                ro_data = nn_f(xf)
                ############################################################
                grad_data = nn_c(xc, ro_data).data
                grad_arr.append( grad_data )
                ############################################################
            # end with
            ################################################################
            ro_arr = [val.reshape(val.shape[0], -1) for val in ro_data]
            x_bool = xp.concatenate(ro_arr, axis=1)
            ################################################################
            if (last_bool is not None):
                x_bool = xp.concatenate([last_bool, x_bool], axis=0)
                grad_data = xp.concatenate([last_grad, grad_data], axis=0)
                l2_val = xp.concatenate([last_l2, l2_val], axis=0)
            # end if
            ################################################################
            ##  normalized by l2 output values
            grad_data = grad_data / l2_val
            ################################################################
            bool0 = x_bool[:-1]
            bool1 = x_bool[1:]
            trans_arr = xp.sum((bool0 != bool1), axis=1)
            ################################################################
            for did, n_trans in enumerate(trans_arr):
                ############################################################
                if (n_trans > 0):
                    ########################################################
                    node_list.append(cpu(n_trans))
                    hess = (grad_data[did+1] - grad_data[did]) / df_val
                    hess_list.append(cpu(hess))
                    ########################################################
                # end if
                ############################################################
            # end for
            ################################################################
            last_bool = x_bool[-1:]
            last_grad = grad_data[-1:]
            last_l2   = l2_val[-1:]
            ################################################################
        # end for
        ####################################################################
        
        ####################################################################
        l2_arr = xp.reshape(xp.asarray(l2_arr), (num_sample, 1))
        ####################################################################
        d_shape = (num_sample, -1)
        ####################################################################
        ## normalized by l2 output values
        grad_arr = xp.concatenate(grad_arr, axis=0).reshape(d_shape) / l2_arr
        grad_arr = cpu(grad_arr) / df_val
        ####################################################################
        
        ####################################################################
        hess_arr = np.array(hess_list).reshape((-1, num_classes))
        node_arr = np.array(node_list).reshape((-1, 1))
        ####################################################################
        switch = node_arr.sum()
        hess_sq = (hess_arr * hess_arr) / node_arr
        hess_sig = np.sqrt(hess_sq.sum(axis=0) / len(node_arr))
        ####################################################################

        ####################################################################
        beg_grd = grad_arr[0]
        end_grd = grad_arr[-1]
        ####################################################################
        ## for mid gradient
        ####################################################################
        mid_switch = switch / 2.0
        sum_switch = 0
        mid_id = None
        mid_mean = False
        ####################################################################
        for did, n_trans in enumerate(node_list):
            ################################################################
            pre_val = mid_switch - sum_switch 
            sum_switch += n_trans
            ################################################################
            if (sum_switch > mid_switch):
                post_val = sum_switch - mid_switch
                if (pre_val < post_val):
                    mid_id = did - 1
                elif (pre_val > post_val):
                    mid_id = did
                elif (pre_val == post_val):
                    mid_id = did
                    mid_mean = True
                # end if
                break
            # end if
            ################################################################
        # end for
        ####################################################################
        if (mid_mean == False):
            mid_grd = grad_arr[mid_id]
        else:
            mid_grd = (grad_arr[mid_id - 1] + grad_arr[mid_id]) / 2
        # end if
        ####################################################################
        linear_mid = (beg_grd + end_grd) / 2
        g_diverge = mid_grd - linear_mid
        g_sigma = g_diverge / (np.sqrt(switch) / 2)
        ####################################################################
        switch_data.append( switch )
        hess_data.extend( hess_sig.tolist() )
        g_sig_data.extend( g_sigma.tolist() )
        ####################################################################
    # end for
    ########################################################################
    print(': end')
    ########################################################################
    # hess_data = np.array(hess_data)
    # switch_data = np.array(switch_data)
    # g_sig_data = np.array(g_sigma)
    ########################################################################
    return hess_data, switch_data, g_sig_data
############################################################################


#############################################################################
def bf_new_mrg(model, train_ds, pair_ids, l2_out, x_div, num_classes, batch):
    #########################################################################
    switch_data = []
    hess_data   = []
    g_sig_data  = []
    marg_data   = []
    ########################################################################
    num_sample = x_div.shape[0]
    all_index = np.arange(num_sample, dtype=np.int)
    ########################################################################
    print('calc. gradient {}:'.format(
        pair_ids.shape[0]), end='', flush=True)
    ########################################################################
    # NN's value
    nn_f = lambda x: model.get_act(x)
    nn_c = lambda x, cut: model.cutoff(x, cut)
    ########################################################################
    for j in range(pair_ids.shape[0]):
        ####################################################################
        ## data setup
        ####################################################################
        t_idx = pair_ids[j][0]
        n_idx = pair_ids[j][1]
        t_img = train_ds[t_idx][0]
        n_img = train_ds[n_idx][0]
        ####################################################################
        grad_arr  = []
        marg_arr  = []
        l2_arr    = []
        node_list = []
        hess_list = []
        ####################################################################
        last_bool = None
        last_grad = None
        last_l2   = None
        ####################################################################
        rl_diff = (t_img - n_img).reshape((-1,))
        ####################################################################
        df_val = np.linalg.norm(rl_diff)
        ####################################################################
        ## check ReLU on off and calc extended gradient data
        ####################################################################
        if (j%10 == 0):
            print('{},'.format(j), end='', flush=True)
        # end if
        ####################################################################
        for i in range(0, num_sample, batch):
            ################################################################
            sub_idx = all_index[i:i + batch]
            ################################################################
            sub_div = x_div[sub_idx]
            x_val = np.array([(1 - t) * t_img + t * n_img for t in sub_div])
            xf = chainer.Variable(xp.asarray(x_val.astype(np.float32)))
            ################################################################
            l2_val =  [(1 - t) * l2_out[t_idx] + t * l2_out[n_idx]
                       for t in sub_div]
            l2_arr.extend( l2_val )
            l2_val = xp.reshape(xp.asarray(l2_val), (-1, 1))
            ################################################################
            x_drc = [ n_img - t_img ] * sub_idx.shape[0]
            xc = chainer.Variable(xp.asarray(x_drc))
            ################################################################
            with chainer.using_config('train', False):
                ############################################################
                ro_data = nn_f(xf)
                marg_arr.append( model.output )
                ############################################################
                grad_data = nn_c(xc, ro_data).data
                grad_arr.append( grad_data )
                ############################################################
            # end with
            ################################################################
            ro_arr = [val.reshape(val.shape[0], -1) for val in ro_data]
            x_bool = xp.concatenate(ro_arr, axis=1)
            ################################################################
            if (last_bool is not None):
                x_bool = xp.concatenate([last_bool, x_bool], axis=0)
                grad_data = xp.concatenate([last_grad, grad_data], axis=0)
                l2_val = xp.concatenate([last_l2, l2_val], axis=0)
            # end if
            ################################################################
            ##  normalized by l2 output values
            grad_data = grad_data / l2_val
            ################################################################
            bool0 = x_bool[:-1]
            bool1 = x_bool[1:]
            trans_arr = xp.sum((bool0 != bool1), axis=1)
            ################################################################
            for did, n_trans in enumerate(trans_arr):
                ############################################################
                if (n_trans > 0):
                    ########################################################
                    node_list.append(cpu(n_trans))
                    hess = (grad_data[did+1] - grad_data[did]) / df_val
                    hess_list.append(cpu(hess))
                    ########################################################
                # end if
                ############################################################
            # end for
            ################################################################
            last_bool = x_bool[-1:]
            last_grad = grad_data[-1:]
            last_l2   = l2_val[-1:]
            ################################################################
        # end for
        ####################################################################
        
        ####################################################################
        l2_arr = xp.reshape(xp.asarray(l2_arr), (num_sample, 1))
        ####################################################################
        d_shape = (num_sample, -1)
        ####################################################################
        ## normalized by l2 output values
        marg_arr = xp.concatenate(marg_arr, axis=0).reshape(d_shape) / l2_arr
        marg_arr = cpu(marg_arr)
        marg_data.append( marg_arr )
        ####################################################################
        ## normalized by l2 output values
        grad_arr = xp.concatenate(grad_arr, axis=0).reshape(d_shape) / l2_arr
        grad_arr = cpu(grad_arr) / df_val
        ####################################################################
        
        ####################################################################
        hess_arr = np.array(hess_list).reshape((-1, num_classes))
        node_arr = np.array(node_list).reshape((-1, 1))
        ####################################################################
        switch = node_arr.sum()
        hess_sq = (hess_arr * hess_arr) / node_arr
        hess_sig = np.sqrt(hess_sq.sum(axis=0) / len(node_arr))
        ####################################################################

        ####################################################################
        beg_grd = grad_arr[0]
        end_grd = grad_arr[-1]
        ####################################################################
        ## for mid gradient
        ####################################################################
        mid_switch = switch / 2.0
        sum_switch = 0
        mid_id = None
        mid_mean = False
        ####################################################################
        for did, n_trans in enumerate(node_list):
            ################################################################
            pre_val = mid_switch - sum_switch 
            sum_switch += n_trans
            ################################################################
            if (sum_switch > mid_switch):
                post_val = sum_switch - mid_switch
                if (pre_val < post_val):
                    mid_id = did - 1
                elif (pre_val > post_val):
                    mid_id = did
                elif (pre_val == post_val):
                    mid_id = did
                    mid_mean = True
                # end if
                break
            # end if
            ################################################################
        # end for
        ####################################################################
        if (mid_mean == False):
            mid_grd = grad_arr[mid_id]
        else:
            mid_grd = (grad_arr[mid_id - 1] + grad_arr[mid_id]) / 2
        # end if
        ####################################################################
        linear_mid = (beg_grd + end_grd) / 2
        g_diverge = mid_grd - linear_mid
        g_sigma = g_diverge / (np.sqrt(switch) / 2)
        ####################################################################
        switch_data.append( switch )
        hess_data.extend( hess_sig.tolist() )
        g_sig_data.extend( g_sigma.tolist() )
        ####################################################################
    # end for
    ########################################################################
    print(': end')
    ########################################################################
    # hess_data = np.array(hess_data)
    # switch_data = np.array(switch_data)
    # g_sig_data = np.array(g_sigma)
    ########################################################################
    return hess_data, switch_data, g_sig_data, marg_data
############################################################################


#############################################################################
def sample_behavior(model, train_ds, t_idx, n_idx, l2_out, num_div):
    #########################################################################
    ## distance of the pair data
    t_img = train_ds[t_idx][0]
    n_img = train_ds[n_idx][0]
    rl_diff = (t_img - n_img).reshape((-1,))
    ########################################################################
    df_val = np.linalg.norm(rl_diff)
    ########################################################################
    x_line = np.linspace(0.0, 1.0, num_div, dtype=np.float)
    #########################################################################
    x_div = {}
    end_flag = False
    count = 0
    key_id = 0
    #########################################################################
    while(end_flag == False):
        if(count + test_bat >= x_line.shape[0]):
            end_flag = True
        # end if
        x_div[key_id] = x_line[count:count + test_bat]
        key_id += 1
        count += test_bat
    # end while
    #########################################################################
    sq_data = []
    sq_grad = []
    #########################################################################
    for x_loc in x_div.values():
        #####################################################################
        x_val = []
        l2_val = []
        for t in x_loc:
            x_val.append((1 - t) * t_img + t * n_img)
            l2_val.append((1 - t) * l2_out[t_idx] + t * l2_out[n_idx])
        # end for
        l2_val = np.asarray(l2_val).reshape((-1, 1))
        x_new = np.asarray(x_val, dtype=np.float32)
        num_sample = x_new.shape[0]
        # NN's value
        nn = lambda x: model.get_act(x)
        #####################################################################
        ## check ReLU on off
        xx = chainer.Variable(xp.asarray(x_new.astype(np.float32)))
        with chainer.using_config('train', False):
            ro_cut = nn(xx)
        # end with
        ####################################################################
        ## normalized by output values
        data_loc = cpu(model.output) / l2_val
        ####################################################################
        # get extend nn_val
        # right side input value
        r_ext = [ t_img ] * num_sample
        r_ext = np.asarray(r_ext, dtype=np.float32)
        # right side input value
        l_ext = [ n_img ] * num_sample
        l_ext = np.asarray(l_ext, dtype=np.float32)
        # NN's value
        nn = lambda x, cut : model.cutoff(x, cut)
        # get nn extended activation
        ####################################################################
        rr = chainer.Variable(xp.asarray(r_ext.astype(np.float32)))
        ll = chainer.Variable(xp.asarray(l_ext.astype(np.float32)))
        ####################################################################
        with chainer.using_config('train', False):
            ################################################################
            ## normalized by output values
            r_val = cpu(nn(rr, ro_cut).data).astype(np.float) / l2_val
            l_val = cpu(nn(ll, ro_cut).data).astype(np.float) / l2_val
            ################################################################
        # end with
        ####################################################################
        grad_loc = (l_val - r_val) / df_val
        ####################################################################
        sq_data.append(data_loc)
        sq_grad.append(grad_loc)
        ####################################################################
    # end for
    ########################################################################
    data_arr = np.concatenate(sq_data, axis=0)
    grad_arr = np.concatenate(sq_grad, axis=0)
    ########################################################################
    return data_arr, grad_arr
#############################################################################
#############################################################################
def calc_node(model, train_ds, t_idx, n_idx, l2_out, x_div):
    #########################################################################
    x_loc = []
    x_key = []
    for key, arr in x_div.items():
        x_key.append(key)
        x_loc.append(arr)
    # end for
    line = np.asarray(x_key).argsort()
    pre_loc = [x_loc[j] for j in line]
    x_loc = np.concatenate(pre_loc).reshape((-1,))
    #########################################################################
    n_flag = True
    #########################################################################
    max_node = 1e-06
    #########################################################################
    x_val = []
    # l2_val = []
    t_img = train_ds[t_idx][0]
    n_img = train_ds[n_idx][0]
    for t in x_loc:
        x_val.append((1 - t) * t_img + t * n_img)
        # l2_val.append((1 - t) * l2_out[t_idx] + t * l2_out[n_idx])
    # end for
    #########################################################################
    # l2_val = np.asarray(l2_val)
    x_arr = np.asarray(x_val, dtype=np.float32)
    num_sample = x_arr.shape[0]
    ro_act = []
    nn = lambda x: model.get_act(x)
    # calc network output
    for i in range(0, num_sample, test_bat):
        #####################################################################
        xx = chainer.Variable(xp.asarray(
            (x_arr[i:i + test_bat]).astype(np.float32)))
        #####################################################################
        with chainer.using_config('train', False):
            ro_data = nn(xx)
        # end with
        #####################################################################
        ro_arr = [val.reshape(val.shape[0], -1) for val in ro_data]
        ro_act.append(cpu(xp.concatenate(ro_arr, axis=1)))
        #####################################################################
    # end for
    #########################################################################
    ## check ReLU on off
    ro_act = np.concatenate(ro_act, axis=0)
    ro_bool = ro_act
    #########################################################################
    ## split the concat data
    #########################################################################
    p_len = []
    x_key = []
    for key, arr in x_div.items():
        x_key.append(key)
        p_len.append(arr.shape[0])
    # end for
    line = np.asarray(x_key).argsort()
    q_len = [p_len[j] for j in line] 
    x_len = []
    total_len = 0
    for val in q_len:
        total_len += val
        x_len.append(total_len)
    # end for
    x_bool = np.split(ro_bool, x_len, axis=0)[:-1]
    #########################################################################
    #########################################################################
    trans_dict = {}
    x_vec = {}
    new_set = {}
    flag_sign = {}
    sign = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    x_div_keys = np.sort(np.asarray(list(x_div))).tolist()
    # print('x_div keys:', x_div_keys)
    for bid, key in enumerate(x_div_keys):
        bool0 = x_bool[bid][:-1]
        bool1 = x_bool[bid][1:]
        trans_dict[key] = (np.sum((bool0 != bool1), axis=1)).tolist()
        x_vec[key] = (x_div[key].reshape((-1,))).tolist()
        new_set[key] = set([x_vec[key][0], x_vec[key][-1]])
        flag_sign[key] = {}
        for sid in sign:
            flag_sign[key][sid] = 0
        # end for
    # end for
    #########################################################################
    x_pos = {}
    #########################################################################
    for j, trans_list in trans_dict.items():
        n_trans = trans_list[0]
        if (n_trans >= 2):
            n_flag = False
            flag_sign[j]['a'] += 1
            n_add = n_trans + 3
            s_arr = np.linspace(x_vec[j][0], x_vec[j][1], n_add, dtype=np.float)
            new_set[j] |= set( (s_arr[1:]).tolist() )
        elif (n_trans == 1):
            n_flag = False
            flag_sign[j]['b'] += 1
            mid = (x_vec[j][0] + x_vec[j][1]) / 2.0
            new_set[j] |= set([mid, x_vec[j][1]])
        else:
            new_set[j] |= set([x_vec[j][1]])
        # end if
        #####################################################################
        if (x_vec[j][1] - x_vec[j][0] > max_node):
            n_flag = False
            flag_sign[j]['f'] += 1
            mid = (x_vec[j][0] + x_vec[j][1]) / 2.0
            new_set[j] |= set([mid])
        # end if
        #####################################################################
        for did in range(1, len(trans_list)):
            n_trans = trans_list[did]
            if (n_trans >= 2):
                if (x_vec[j][did] in new_set[j]):
                    mid = (x_vec[j][did-1] + x_vec[j][did]) / 2.0
                    new_set[j] |= set([mid])
                # end if
                n_flag = False
                flag_sign[j]['c'] += 1
                n_add = n_trans + 3
                s_arr = np.linspace(
                    x_vec[j][did], x_vec[j][did+1], n_add, dtype=np.float)
                new_set[j] |= set( s_arr.tolist() )
            elif (n_trans == 1):
                if (x_vec[j][did] in new_set[j]):
                    n_flag = False
                    flag_sign[j]['d'] += 1
                    mid_0 = (x_vec[j][did-1] + x_vec[j][did]) / 2.0
                    mid_1 = (x_vec[j][did] + x_vec[j][did+1]) / 2.0
                    new_set[j] |= set([mid_0, mid_1])
                # end if
                if (x_vec[j][did+1] - x_vec[j][did] > max_node):
                    n_flag = False
                    flag_sign[j]['g'] += 1
                    mid = (x_vec[j][did] + x_vec[j][did+1]) / 2.0
                    new_set[j] |= set([mid])
                # end if
                new_set[j] |= set([x_vec[j][did], x_vec[j][did+1]])
            # end if
        # end for
        #####################################################################
        new_set[j] |= set([x_vec[j][-2]])
        #####################################################################
        if (trans_list[-1] == 1):
            n_flag = False
            flag_sign[j]['e'] += 1
            mid = (x_vec[j][-2] + x_vec[j][-1]) / 2.0
            new_set[j] |= set([mid])
        # end if
        #####################################################################
        if (x_vec[j][-1] - x_vec[j][-2] > max_node):
            n_flag = False
            flag_sign[j]['h'] += 1
            mid = (x_vec[j][-2] + x_vec[j][-1]) / 2.0
            new_set[j] |= set([mid])
        # end if
        #####################################################################
        x_pre = list(new_set[j])
        x_pre = sorted(x_pre)
        x_pos[j] = np.asarray(x_pre, dtype=np.float)
        #####################################################################
    # end for
    #########################################################################
    return n_flag, flag_sign, x_pos
#############################################################################
#############################################################################
def search_div(model, train_ds, t_idx, n_idx, l2_out, x_div):
    #########################################################################
    x_spt = {}
    #########################################################################
    n_flag = False
    count = 0
    c_num = {}
    d_num = {}
    g_num = {}
    x_num = {}
    for j in x_div.keys():
        c_num[j] = None
        d_num[j] = None
        g_num[j] = None
        x_num[j] = None
    # end for
    # print('start calculation for gradient')
    #########################################################################
    while(n_flag == False):
        #####################################################################
        count += 1
        #####################################################################
        n_flag, flag_sign, x_div = calc_node(
            model, train_ds, t_idx, n_idx, l2_out, x_div)
        #####################################################################
        j_all = 0
        c_all = 0
        d_all = 0
        g_all = 0
        x_all = 0
        #####################################################################
        x_div_keys = list(x_div)
        for j in x_div_keys:
            #################################################################
            if ( (c_num[j] == flag_sign[j]['c']) and
                 (d_num[j] == flag_sign[j]['d']) and
                 (g_num[j] == flag_sign[j]['g']) and
                 (x_num[j] == x_div[j].shape[0]) ):
                if (c_num[j] == 0) or (d_num[j] == 0):
                    x_spt[j] = x_div.pop(j)
                # end if
            # end if
            #################################################################
            if (j in x_div):
                min_node = (x_div[j][1:] - x_div[j][:-1]).min()
                if( min_node < sys.float_info.epsilon ):
                    x_spt[j] = x_div.pop(j)
                # end if
            # end if
            #################################################################
            if (j in x_div):
                j_all += 1
                c_num[j] = flag_sign[j]['c']
                d_num[j] = flag_sign[j]['d']
                g_num[j] = flag_sign[j]['g']
                x_num[j] = x_div[j].shape[0]
                c_all += c_num[j]
                d_all += d_num[j]
                g_all += g_num[j]
                x_all += x_num[j]
            # end if
            #################################################################
        # end for
        #####################################################################
        if (count > 32) or (n_flag == True) or (j_all == 0):
            x_div_keys = list(x_div)
            for j in x_div_keys:
                x_spt[j] = x_div.pop(j)
            # end for
            n_flag = True
        # end if
        #####################################################################
        print('{},'.format(j_all), end='', flush=True)
        # print('{},'.format(x_all), end='', flush=True)
        # print('search :count =', count, ', j =', j_all, ', c =', c_all,
        #       ', d =', d_all, ', g =', g_all, ' ,x =', x_all)
        #####################################################################
    # end while
    #########################################################################
    return x_spt
#############################################################################
#############################################################################
def get_site_div(model, train_ds, t_idx, n_idx, l2_out,
                 init_size=1.0, n_split=1, n_refer=512, ex_rate=1):
    #########################################################################
    site0 = 0.5 - init_size * 0.5
    site1 = 0.5 + init_size * 0.5
    x_info = {}
    x_info[0] = np.linspace(site0, site1, 2048, dtype=np.float)
    #########################################################################
    n_flag, flag_sign, x_info = calc_node(
        model, train_ds, t_idx, n_idx, l2_out, x_info)
    #########################################################################
    n_node = x_info[0].shape[0] / init_size
    s_size = n_refer / n_node
    #########################################################################
    if (s_size < 1.0):
        n_div = int(n_node * s_size)
    else:
        s_size = 1.0
        n_div = 2048
    # end if
    #########################################################################
    if (s_size * n_split > 0.5):
        s_size = 1.0
        n_div = 2048
    # end if
    #########################################################################
    pre_div = {}
    pre_site = {}
    #########################################################################
    if (s_size < 1.0):
        w_site = 1.0 / n_split
        for j in range(n_split):
            pre_site[j] = [w_site * j, w_site * j + s_size]
            pre_div[j] = np.linspace(
                pre_site[j][0], pre_site[j][1], n_div, dtype=np.float)
        # end for
    else:
        pre_site[0] = [0.0, 1.0]
        pre_div[0] = np.linspace(0.0, 1.0, 2048, dtype=np.float)
    # end if
    #########################################################################
    #########################################################################
    print('search ', end='', flush=True)
    x_spt = search_div(model, train_ds, t_idx, n_idx, l2_out, pre_div)
    print(' end')
    #########################################################################
    #########################################################################
    x_all = 0
    s_all = 0.0
    for j in x_spt.keys():
        x_all += x_spt[j].shape[0]
        s_all += pre_site[j][1] - pre_site[j][0]
    # end for
    #########################################################################
    n_node = x_all / s_all
    s_size = n_refer / n_node
    #########################################################################
    if (s_size < 1.0):
        n_div = int(n_node * s_size)
    else:
        s_size = 1.0
        n_div = 2048
        n_split = 1
    # end if
    #########################################################################
    if (s_size * n_split > 0.5):
        s_size = 1.0
        n_div = 2048
        n_split = 1
    # end if
    #########################################################################
    x_div = {}
    x_site = {}
    #########################################################################
    if (s_size < 1.0):
        #####################################################################
        w_site = 1.0 / n_split
        for j in range(n_split):
            x_site[j] = [w_site * j, w_site * j + s_size]
            x_div[j]= np.linspace(
                x_site[j][0], x_site[j][1], n_div * ex_rate, dtype=np.float)
        # end for
        #####################################################################
    else:
        #####################################################################
        x_site[0] = [0.0, 1.0]
        x_div[0] = np.linspace(0.0, 1.0, 2048, dtype=np.float)
        #####################################################################
    # end if
    #########################################################################
    return x_site, x_div, s_size * n_split
#############################################################################
#############################################################################
def bf_hess(model, train_ds, t_idx, n_idx, l2_out, num_classes, num_acts):
    #########################################################################
    # get relu profile
    #########################################################################
    x_div = np.linspace(0.0, 1.0, num_acts, dtype=np.float)
    #########################################################################
    x_val = []
    l2_val = []
    t_img = train_ds[t_idx][0]
    n_img = train_ds[n_idx][0]
    for t in x_div:
        x_val.append((1 - t) * t_img + t * n_img)
        l2_val.append((1 - t) * l2_out[t_idx] + t * l2_out[n_idx])
    # end for
    l2_val = np.asarray(l2_val).reshape((-1, 1))
    x_new = np.asarray(x_val, dtype=np.float32)
    num_sample = x_new.shape[0]
    ########################################################################
    # NN's value
    nn = lambda x: model.get_act(x)
    ########################################################################
    ## check ReLU on off
    ########################################################################
    ro_new = []
    ro_cut = {}
    #########################################################################
    for i in range(0, num_sample, test_bat):
        ####################################################################
        xx = chainer.Variable(xp.asarray(
            (x_new[i:i + test_bat]).astype(np.float32)))
        ####################################################################
        with chainer.using_config('train', False):
            ro_data = nn(xx)
        # end with
        ####################################################################
        ro_cut[i] = ro_data
        ####################################################################
        ro_arr = [val.reshape(val.shape[0], -1) for val in ro_data]
        ro_new.append(cpu(xp.concatenate(ro_arr, axis=1)))
        ####################################################################
    # end for
    ########################################################################
    ro_new = np.concatenate(ro_new, axis=0)
    ########################################################################
    x_bool = ro_new
    bool0 = x_bool[:-1]
    bool1 = x_bool[1:]
    trans_list = (np.sum((bool0 != bool1), axis=1)).tolist()
    ########################################################################

    
    #########################################################################
    ##  calc extended gradient data
    ########################################################################
    # NN's value
    nn = lambda x, cut: model.cutoff(x, cut)
    #########################################################################
    x_drc = np.asarray([ n_img - t_img ] * num_sample)
    ########################################################################
    out_data = []
    ########################################################################
    for i in range(0, num_sample, test_bat):
        ####################################################################
        xx = chainer.Variable(xp.asarray(x_drc[i:i + test_bat]))
        act = ro_cut[i]
        ####################################################################
        with chainer.using_config('train', False):
            ext_val = nn(xx, act)
        # end with
        ####################################################################
        out_data.append(cpu(ext_val.data))
        ####################################################################
    # end for
    ########################################################################
    ## normalized by output values
    out_data = np.concatenate(out_data, axis=0) / l2_val
    ########################################################################

    
    ########################################################################
    # get Hessian data
    ########################################################################
    rl_diff = (t_img - n_img).reshape((-1,))
    df_val = np.linalg.norm(rl_diff)
    ########################################################################
    sub_list  = []
    node_list = []
    all_list  = []
    ########################################################################
    for did, n_trans in enumerate(trans_list):
        ####################################################################
        if (n_trans == 1):
            ################################################################
            hess = (out_data[did+1] - out_data[did]) / df_val
            sub_list.append(hess)
            ################################################################
        # end if
        ####################################################################
        if (n_trans > 0):
            ################################################################
            node_list.append(n_trans)
            hess = (out_data[did+1] - out_data[did]) / df_val
            all_list.append(hess)
            ################################################################
        # end if
        ####################################################################
    # end for
    ########################################################################
    sub_arr  = np.array(sub_list, dtype=np.float)
    all_arr  = np.array(all_list, dtype=np.float).reshape((-1, num_classes))
    node_arr = np.array(node_list, dtype=np.float).reshape((-1, 1))
    ########################################################################
    sub_sig  = sub_arr.std()
    ########################################################################
    switch = node_arr.sum()
    hess_sq = (all_arr * all_arr) / node_arr
    hess_sig = np.sqrt(hess_sq.sum(axis=0) / len(node_arr))
    ########################################################################
    beg_grd = out_data[0] / df_val
    end_grd = out_data[-1] / df_val
    ########################################################################
    ## for mid gradient
    ########################################################################
    mid_switch = switch / 2.0
    sum_switch = 0
    mid_id = None
    mid_mean = False
    for did, n_trans in enumerate(node_list):
        pre_val = mid_switch - sum_switch 
        sum_switch += n_trans
        if (sum_switch > mid_switch):
            post_val = sum_switch - mid_switch
            if (pre_val < post_val):
                mid_id = did - 1
            elif (pre_val > post_val):
                mid_id = did
            elif (pre_val == post_val):
                mid_id = did
                mid_mean = True
            # end if
            break
        # end if
    # end for
    if (mid_mean == False):
        mid_grd = out_data[mid_id] / df_val
    else:
        mid_grd = (out_data[mid_id - 1] + out_data[mid_id]) / (2 * df_val)
    # end if
    mgn_grd = out_data[int(num_sample/2)] / df_val
    ########################################################################
    # print('switch, node =', switch, len(node_arr))
    ########################################################################
    return hess_sig, switch, beg_grd, end_grd, mid_grd, sub_sig, mgn_grd
############################################################################


#############################################################################
def bf_margin(model, train_ds, pair_ids, l2_out, x_div, num_classes, num_acts):
    #########################################################################
    # data setup
    #########################################################################
    print(' data setup :', end='', flush=True)
    #########################################################################
    x_val = []
    x_drc = []
    l2_val = []
    #########################################################################
    for j in range(pair_ids.shape[0]):
        #####################################################################
        t_idx = pair_ids[j][0]
        n_idx = pair_ids[j][1]
        t_img = train_ds[t_idx][0]
        n_img = train_ds[n_idx][0]
        l2_bit = []
        #####################################################################
        for t in x_div:
            x_val.append((1 - t) * t_img + t * n_img)
            l2_bit.append(l2_out[t_idx] * (1-t) +  l2_out[n_idx] * t)
        # end for
        #####################################################################
        l2_val.append(l2_bit)
        #####################################################################
        x_drc.extend([ n_img - t_img ] * x_div.shape[0])
        #####################################################################
    # end for
    ########################################################################
    ## l2 output values for normalization
    d_shape = (pair_ids.shape[0], x_div.shape[0], 1)
    l2_val = xp.reshape(xp.asarray(l2_val), d_shape)
    ########################################################################
    x_new = np.asarray(x_val, dtype=np.float32)
    num_sample = x_new.shape[0]
    ########################################################################

    
    ########################################################################
    ## check ReLU on off and calc margin and extended gradient data
    ########################################################################
    print(' calc. margin and grad :', end='', flush=True)
    ########################################################################
    # NN's value
    nn_f = lambda x: model.get_act(x)
    nn_c = lambda x, cut: model.cutoff(x, cut)
    ########################################################################
    mrg_data = []
    out_data = []
    ro_new = []
    #########################################################################
    for i in range(0, num_sample, test_bat):
        ####################################################################
        xf = chainer.Variable(xp.asarray(
            (x_new[i:i + test_bat]).astype(np.float32)))
        xc = chainer.Variable(xp.asarray(x_drc[i:i + test_bat]))
        ####################################################################
        with chainer.using_config('train', False):
            ################################################################
            ro_data = nn_f(xf)
            mrg_data.append(model.output)
            ################################################################
            out_data.append( nn_c(xc, ro_data).data )
            ################################################################
        # end with
        ####################################################################
        ro_arr = [val.reshape(val.shape[0], -1) for val in ro_data]
        ro_new.append(cpu(xp.concatenate(ro_arr, axis=1)))
        ####################################################################
    # end for
    ########################################################################
    d_shape = (pair_ids.shape[0], x_div.shape[0], -1)
    ########################################################################
    ## normalized by l2 output values
    mrg_data = xp.concatenate(mrg_data, axis=0).reshape(d_shape)
    mrg_data = cpu(mrg_data / l2_val)
    ########################################################################
    ## normalized by l2 output values
    out_data = xp.concatenate(out_data, axis=0).reshape(d_shape)
    out_data = cpu(out_data / l2_val)
    ########################################################################
    ro_new = np.concatenate(ro_new, axis=0)
    x_bool = ro_new.reshape(d_shape)
    ########################################################################

    
    ########################################################################
    # split pair data
    ########################################################################
    trans_dict = {}
    out_dict = {}
    ########################################################################
    for key in range(pair_ids.shape[0]):
        bool0 = x_bool[key][:-1]
        bool1 = x_bool[key][1:]
        trans_dict[key] = (np.sum((bool0 != bool1), axis=1)).tolist()
        out_dict[key]  = out_data[key]
    # end for
    ########################################################################
    

    ########################################################################
    # get Hessian data
    ########################################################################
    print(' calc. hessian :', end='', flush=True)
    ########################################################################
    hess_data = []
    switch_data = []
    g_sig_data = []
    ########################################################################
    for j, trans_list in trans_dict.items():
        ####################################################################
        t_idx = pair_ids[j][0]
        n_idx = pair_ids[j][1]
        t_img = train_ds[t_idx][0]
        n_img = train_ds[n_idx][0]
        rl_diff = (t_img - n_img).reshape((-1,))
        df_val = np.linalg.norm(rl_diff)
        ####################################################################
        
        ####################################################################
        node_list = []
        hess_list  = []
        ####################################################################
        for did, n_trans in enumerate(trans_list):
            ################################################################
            if (n_trans > 0):
                ############################################################
                node_list.append(n_trans)
                hess = (out_dict[j][did+1] - out_dict[j][did]) / df_val
                hess_list.append(hess)
                ############################################################
            # end if
            ################################################################
        # end for
        ####################################################################
        hess_arr = np.array(hess_list).reshape((-1, num_classes))
        node_arr = np.array(node_list).reshape((-1, 1))
        ####################################################################
        switch = node_arr.sum()
        hess_sq = (hess_arr * hess_arr) / node_arr
        hess_sig = np.sqrt(hess_sq.sum(axis=0) / len(node_arr))
        ####################################################################

        ####################################################################
        beg_grd = out_dict[j][0] / df_val
        end_grd = out_dict[j][-1] / df_val
        ####################################################################
        ## for mid gradient
        ####################################################################
        mid_switch = switch / 2.0
        sum_switch = 0
        mid_id = None
        mid_mean = False
        ####################################################################
        for did, n_trans in enumerate(node_list):
            ################################################################
            pre_val = mid_switch - sum_switch 
            sum_switch += n_trans
            ################################################################
            if (sum_switch > mid_switch):
                post_val = sum_switch - mid_switch
                if (pre_val < post_val):
                    mid_id = did - 1
                elif (pre_val > post_val):
                    mid_id = did
                elif (pre_val == post_val):
                    mid_id = did
                    mid_mean = True
                # end if
                break
            # end if
            ################################################################
        # end for
        ####################################################################
        if (mid_mean == False):
            mid_grd = out_dict[j][mid_id] / df_val
        else:
            mid_val = (out_dict[j][mid_id - 1] + out_dict[j][mid_id]) / 2
            mid_grd = mid_val / df_val
        # end if
        ####################################################################
        linear_mid = (beg_grd + end_grd) / 2
        g_diverge = mid_grd - linear_mid
        g_sigma = g_diverge / (np.sqrt(switch) / 2)
        ####################################################################
        switch_data.append(switch)
        hess_data.extend(hess_sig.tolist())
        g_sig_data.extend(g_sigma.tolist())
        ####################################################################
    # end for
    ########################################################################
    print(' :end')
    ########################################################################
    # hess_data = np.array(hess_data)
    # switch_data = np.array(switch_data)
    # g_sig_data = np.array(g_sigma)
    ########################################################################
    return hess_data, switch_data, g_sig_data, mrg_data
############################################################################


#############################################################################
def bf_calc(model, train_ds, pair_ids, l2_out, x_div, num_classes, num_acts):
    #########################################################################
    # data setup
    #########################################################################
    x_val = []
    x_drc = []
    l2_val = []
    #########################################################################
    for j in range(pair_ids.shape[0]):
        #####################################################################
        t_idx = pair_ids[j][0]
        n_idx = pair_ids[j][1]
        t_img = train_ds[t_idx][0]
        n_img = train_ds[n_idx][0]
        l2_bit = []
        #####################################################################
        for t in x_div:
            x_val.append((1 - t) * t_img + t * n_img)
            l2_bit.append(l2_out[t_idx] * (1-t) +  l2_out[n_idx] * t)
        # end for
        ####################################################################
        l2_val.append(l2_bit)
        #####################################################################
        x_drc.extend([ n_img - t_img ] * x_div.shape[0])
        ####################################################################
    # end for
    ########################################################################
    ## l2 output values for normalization
    d_shape = (pair_ids.shape[0], x_div.shape[0], 1)
    l2_val = xp.reshape(xp.asarray(l2_val), d_shape)
    ########################################################################
    x_new = np.asarray(x_val, dtype=np.float32)
    num_sample = x_new.shape[0]
    ########################################################################


    ########################################################################
    ## check ReLU on off and calc extended gradient data
    ########################################################################
    # NN's value
    nn_f = lambda x: model.get_act(x)
    nn_c = lambda x, cut: model.cutoff(x, cut)
    ########################################################################
    ro_new = []
    out_data = []
    #########################################################################
    for i in range(0, num_sample, test_bat):
        ####################################################################
        xf = chainer.Variable(xp.asarray(
            (x_new[i:i + test_bat]).astype(np.float32)))
        xc = chainer.Variable(xp.asarray(x_drc[i:i + test_bat]))
        ####################################################################
        with chainer.using_config('train', False):
            ################################################################
            ro_data = nn_f(xf)
            ################################################################
            out_data.append( nn_c(xc, ro_data).data )
            ################################################################
        # end with
        ####################################################################
        ro_arr = [val.reshape(val.shape[0], -1) for val in ro_data]
        ro_new.append(cpu(xp.concatenate(ro_arr, axis=1)))
        ####################################################################
    # end for
    ########################################################################
    d_shape = (pair_ids.shape[0], x_div.shape[0], -1)
    ########################################################################
    ## normalized by l2 output values
    out_data = xp.concatenate(out_data, axis=0).reshape(d_shape)
    out_data = cpu(out_data / l2_val)
    ########################################################################
    ro_new = np.concatenate(ro_new, axis=0)
    x_bool = ro_new.reshape(d_shape)
    ########################################################################

    
    ########################################################################
    # split pair data
    ########################################################################
    trans_dict = {}
    out_dict = {}
    ########################################################################
    for key in range(pair_ids.shape[0]):
        bool0 = x_bool[key][:-1]
        bool1 = x_bool[key][1:]
        trans_dict[key] = (np.sum((bool0 != bool1), axis=1)).tolist()
        out_dict[key]  = out_data[key]
    # end for
    ########################################################################


    ########################################################################
    # get Hessian data
    ########################################################################
    hess_data = []
    switch_data = []
    g_sig_data = []
    ########################################################################
    for j, trans_list in trans_dict.items():
        ####################################################################
        t_idx = pair_ids[j][0]
        n_idx = pair_ids[j][1]
        t_img = train_ds[t_idx][0]
        n_img = train_ds[n_idx][0]
        rl_diff = (t_img - n_img).reshape((-1,))
        df_val = np.linalg.norm(rl_diff)
        ####################################################################
        
        ####################################################################
        node_list = []
        hess_list  = []
        ####################################################################
        for did, n_trans in enumerate(trans_list):
            ################################################################
            if (n_trans > 0):
                ############################################################
                node_list.append(n_trans)
                hess = (out_dict[j][did+1] - out_dict[j][did]) / df_val
                hess_list.append(hess)
                ############################################################
            # end if
            ################################################################
        # end for
        ####################################################################
        hess_arr = np.array(hess_list).reshape((-1, num_classes))
        node_arr = np.array(node_list).reshape((-1, 1))
        ####################################################################
        switch = node_arr.sum()
        hess_sq = (hess_arr * hess_arr) / node_arr
        hess_sig = np.sqrt(hess_sq.sum(axis=0) / len(node_arr))
        ####################################################################

        ####################################################################
        beg_grd = out_dict[j][0] / df_val
        end_grd = out_dict[j][-1] / df_val
        ####################################################################
        ## for mid gradient
        ####################################################################
        mid_switch = switch / 2.0
        sum_switch = 0
        mid_id = None
        mid_mean = False
        ####################################################################
        for did, n_trans in enumerate(node_list):
            ################################################################
            pre_val = mid_switch - sum_switch 
            sum_switch += n_trans
            ################################################################
            if (sum_switch > mid_switch):
                post_val = sum_switch - mid_switch
                if (pre_val < post_val):
                    mid_id = did - 1
                elif (pre_val > post_val):
                    mid_id = did
                elif (pre_val == post_val):
                    mid_id = did
                    mid_mean = True
                # end if
                break
            # end if
            ################################################################
        # end for
        ####################################################################
        if (mid_mean == False):
            mid_grd = out_dict[j][mid_id] / df_val
        else:
            mid_val = (out_dict[j][mid_id - 1] + out_dict[j][mid_id]) / 2
            mid_grd = mid_val / df_val
        # end if
        ####################################################################
        linear_mid = (beg_grd + end_grd) / 2
        g_diverge = mid_grd - linear_mid
        g_sigma = g_diverge / (np.sqrt(switch) / 2)
        ####################################################################
        switch_data.append(switch)
        hess_data.extend(hess_sig.tolist())
        g_sig_data.extend(g_sigma.tolist())
        ####################################################################
    # end for
    ########################################################################
    # hess_data = np.array(hess_data)
    # switch_data = np.array(switch_data)
    # g_sig_data = np.array(g_sigma)
    ########################################################################
    return hess_data, switch_data, g_sig_data
############################################################################


#############################################################################
def sw_mlp_calc(model, train_ds, pair_ids, x_div, num_classes, num_acts):
    #########################################################################
    # data setup
    #########################################################################
    x_val = []
    #########################################################################
    for j in range(pair_ids.shape[0]):
        t_idx = pair_ids[j][0]
        n_idx = pair_ids[j][1]
        t_img = train_ds[t_idx][0]
        n_img = train_ds[n_idx][0]
        for t in x_div:
            x_val.append((1 - t) * t_img + t * n_img)
        # end for
    # end for
    x_new = np.asarray(x_val, dtype=np.float32)
    num_sample = x_new.shape[0]
    ########################################################################

    
    ########################################################################
    ## search ReLU on off
    ########################################################################
    # NN's value
    nn_f = lambda x: model.get_act(x)
    ########################################################################
    ro_acts = []
    #########################################################################
    for i in range(0, num_sample, test_bat):
        ####################################################################
        xf = chainer.Variable(xp.asarray(
            (x_new[i:i + test_bat]).astype(np.float32)))
        ####################################################################
        with chainer.using_config('train', False):
            ro_data = nn_f(xf)
        # end with
        ####################################################################
        ro_arr = [val.reshape(val.shape[0], -1) for val in ro_data]
        ro_acts.append(cpu(xp.concatenate(ro_arr, axis=1)))
        ####################################################################
    # end for
    ########################################################################
    ro_acts = np.concatenate(ro_acts, axis=0)
    ########################################################################
    layer_shape = (num_sample, len(num_acts), num_acts[0])
    ro_layer = ro_acts.reshape(layer_shape)
    ########################################################################

    
    ########################################################################
    # split pair data
    ########################################################################
    x_len = []
    total_len = 0
    num_div = x_div.shape[0]
    for j in range(pair_ids.shape[0]):
        total_len += num_div
        x_len.append(total_len)
    # end for
    ########################################################################
    x_bool = np.split(ro_acts, x_len, axis=0)[:-1]
    l_bool = np.split(ro_layer, x_len, axis=0)[:-1]
    ########################################################################
    trans_dict = {}
    layer_dict = {}
    ########################################################################
    for key in range(pair_ids.shape[0]):
        ####################################################################
        bool0 = x_bool[key][:-1]
        bool1 = x_bool[key][1:]
        trans_dict[key] = (np.sum((bool0 != bool1), axis=1)).tolist()
        ####################################################################
        l_bool0 = l_bool[key][:-1]
        l_bool1 = l_bool[key][1:]
        layer_dict[key] = (np.sum((l_bool0 != l_bool1), axis=2)).tolist()
        ####################################################################
    # end for
    ########################################################################


    ########################################################################
    # get Hessian data
    ########################################################################
    switch_data = []
    ########################################################################
    for j, trans_list in trans_dict.items():
        ####################################################################
        node_list = []
        sw_count = 0
        ####################################################################
        for did, n_trans in enumerate(trans_list):
            ################################################################
            if (n_trans > 0):
                ############################################################
                node_list.append(layer_dict[j][did])
                sw_count += 1
                ############################################################
            # end if
            ################################################################
        # end for
        ####################################################################
        node_arr = np.asarray(node_list).reshape((sw_count, len(num_acts)))
        ####################################################################
        switch = np.sum(node_arr, axis=0)
        ####################################################################
        switch_data.append(switch)
        ####################################################################
    # end for
    ########################################################################
    return switch_data
############################################################################


#############################################################################
def bforce_hess(model, train_ds, t_idx, n_idx, l2_out, x_div,
                num_classes, normalize=True):
    #########################################################################
    # get relu profile and nn_val at nodes
    #########################################################################
    width_list = []
    p_loc = []
    x_key = []
    for key, arr in x_div.items():
        x_key.append(key)
        p_loc.append(arr)
        width_list.append(arr[-1] - arr[0])
    # end for
    width = np.asarray(width_list).sum()
    line = np.asarray(x_key).argsort()
    q_loc = [p_loc[j] for j in line]
    x_loc = np.concatenate(q_loc).reshape((-1,))
    #########################################################################
    x_val = []
    l2_val = []
    t_img = train_ds[t_idx][0]
    n_img = train_ds[n_idx][0]
    for t in x_loc:
        x_val.append((1 - t) * t_img + t * n_img)
        l2_val.append((1 - t) * l2_out[t_idx] + t * l2_out[n_idx])
    # end for
    l2_val = np.asarray(l2_val).reshape((-1, 1))
    x_new = np.asarray(x_val, dtype=np.float32)
    num_sample = x_new.shape[0]
    # NN's value
    nn = lambda x: model.get_act(x)
    #########################################################################
    ## check ReLU on off
    ro_new = []
    ro_cut = {}
    for i in range(0, num_sample, test_bat):
        ####################################################################
        xx = chainer.Variable(xp.asarray(
            (x_new[i:i + test_bat]).astype(np.float32)))
        ####################################################################
        with chainer.using_config('train', False):
            ro_data = nn(xx)
        # end with
        ####################################################################
        ro_cut[i] = ro_data
        ####################################################################
        ro_arr = [val.reshape(val.shape[0], -1) for val in ro_data]
        ro_new.append(cpu(xp.concatenate(ro_arr, axis=1)))
        ####################################################################
    # end for
    ########################################################################
    ro_new = np.concatenate(ro_new, axis=0)
    ########################################################################
    # get extend nn_val
    # right side input value
    r_ext = [ t_img ] * num_sample
    r_ext = np.asarray(r_ext, dtype=np.float32)
    # right side input value
    l_ext = [ n_img ] * num_sample
    l_ext = np.asarray(l_ext, dtype=np.float32)
    # NN's value
    nn = lambda x, cut : model.cutoff(x, cut)
    # get nn extended activation
    r_ext_val = []
    l_ext_val = []
    ########################################################################
    for i in range(0, num_sample, test_bat):
        ####################################################################
        rr = chainer.Variable(xp.asarray(
            (r_ext[i:i + test_bat]).astype(np.float32)))
        ll = chainer.Variable(xp.asarray(
            (l_ext[i:i + test_bat]).astype(np.float32)))
        ####################################################################
        act = ro_cut[i]
        ####################################################################
        with chainer.using_config('train', False):
            r_nn = nn(rr, act)
            l_nn = nn(ll, act)
        # end with
        r_ext_val.append(cpu(r_nn.data))
        l_ext_val.append(cpu(l_nn.data))
    # end for
    ########################################################################
    ## normalized by output values
    r_val = np.concatenate(r_ext_val, axis=0).astype(np.float) / l2_val
    l_val = np.concatenate(l_ext_val, axis=0).astype(np.float) / l2_val
    ########################################################################
    ## node count
    rl_diff = (t_img - n_img).reshape((-1,))
    ########################################################################
    df_val = np.linalg.norm(rl_diff)
    if (normalize == False):
        df_val = 1.0
    # end if
    ########################################################################
    # split concat data
    ########################################################################
    p_len = []
    x_key = []
    for key, arr in x_div.items():
        x_key.append(key)
        p_len.append(arr.shape[0])
    # end for
    line = np.asarray(x_key).argsort()
    q_len = [p_len[j] for j in line] 
    x_len = []
    total_len = 0
    for val in q_len:
        total_len += val
        x_len.append(total_len)
    # end for
    ########################################################################
    new_bool = ro_new
    x_bool = np.split(new_bool, x_len, axis=0)[:-1]
    ########################################################################
    l_split  = np.split(l_val, x_len, axis=0)[:-1]
    r_split  = np.split(r_val, x_len, axis=0)[:-1]
    ########################################################################
    trans_dict = {}
    l_dict = {}
    r_dict = {}
    x_div_keys = list(x_div)
    for key in x_div_keys:
        bool0 = x_bool[key][:-1]
        bool1 = x_bool[key][1:]
        trans_dict[key] = (np.sum((bool0 != bool1), axis=1)).tolist()
        l_dict[key]  = l_split[key]
        r_dict[key]  = r_split[key]
    # end for
    ########################################################################
    # get extended gradient data
    ########################################################################
    hess_list = []
    node = 0
    ########################################################################
    for j, trans_list in trans_dict.items():
        ####################################################################
        for did, n_trans in enumerate(trans_list):
            ################################################################
            # if (n_trans == 1):
            if (n_trans > 0):
                ############################################################
                dv_0 = (l_dict[j][did] - r_dict[j][did]) / df_val
                dv_1 = (l_dict[j][did + 1] - r_dict[j][did + 1]) / df_val
                ############################################################
                hess_list.append(dv_1 - dv_0)
                node += 1
                ############################################################
            # end if
            ################################################################
        # end for
        ####################################################################
    # end for
    ########################################################################
    rev_node = node / width
    hess_arr = np.array(hess_list, dtype=np.float).reshape((-1, num_classes))
    # print('div:', num_sample, ', hess_arr shape:', hess_arr.shape)
    ########################################################################
    return hess_arr, df_val, rev_node
#############################################################################
#############################################################################
def extend_hess(model, train_ds, t_idx, n_idx, l2_out, x_div,
                num_classes, normalize=True):
    #########################################################################
    print('search:', end='', flush=True)
    x_spt = search_div(model, train_ds, t_idx, n_idx, l2_out, x_div)
    print(':end')
    #########################################################################
    # get relu profile and nn_val at nodes
    #########################################################################
    p_loc = []
    x_key = []
    for key, arr in x_spt.items():
        x_key.append(key)
        p_loc.append(arr)
    # end for
    line = np.asarray(x_key).argsort()
    q_loc = [p_loc[j] for j in line]
    x_loc = np.concatenate(q_loc).reshape((-1,))
    #########################################################################
    x_val = []
    l2_val = []
    t_img = train_ds[t_idx][0]
    n_img = train_ds[n_idx][0]
    for t in x_loc:
        x_val.append((1 - t) * t_img + t * n_img)
        l2_val.append((1 - t) * l2_out[t_idx] + t * l2_out[n_idx])
    # end for
    l2_val = np.asarray(l2_val).reshape((-1, 1))
    x_new = np.asarray(x_val, dtype=np.float32)
    num_sample = x_new.shape[0]
    # NN's value
    nn = lambda x: model.get_act(x)
    #########################################################################
    ## check ReLU on off
    ro_new = []
    ro_cut = {}
    for i in range(0, num_sample, test_bat):
        ####################################################################
        xx = chainer.Variable(xp.asarray(
            (x_new[i:i + test_bat]).astype(np.float32)))
        ####################################################################
        with chainer.using_config('train', False):
            ro_data = nn(xx)
        # end with
        ####################################################################
        ro_cut[i] = ro_data
        ####################################################################
        ro_arr = [val.reshape(val.shape[0], -1) for val in ro_data]
        ro_new.append(cpu(xp.concatenate(ro_arr, axis=1)))
        ####################################################################
    # end for
    ########################################################################
    ro_new = np.concatenate(ro_new, axis=0)
    ########################################################################
    # get extend nn_val
    # right side input value
    r_ext = [ t_img ] * num_sample
    r_ext = np.asarray(r_ext, dtype=np.float32)
    # right side input value
    l_ext = [ n_img ] * num_sample
    l_ext = np.asarray(l_ext, dtype=np.float32)
    # NN's value
    nn = lambda x, cut : model.cutoff(x, cut)
    # get nn extended activation
    r_ext_val = []
    l_ext_val = []
    ########################################################################
    for i in range(0, num_sample, test_bat):
        ####################################################################
        rr = chainer.Variable(xp.asarray(
            (r_ext[i:i + test_bat]).astype(np.float32)))
        ll = chainer.Variable(xp.asarray(
            (l_ext[i:i + test_bat]).astype(np.float32)))
        ####################################################################
        cut_arr = ro_cut[i]
        ####################################################################
        with chainer.using_config('train', False):
            r_nn = nn(rr, cut_arr)
            l_nn = nn(ll, cut_arr)
        # end with
        r_ext_val.append(cpu(r_nn.data))
        l_ext_val.append(cpu(l_nn.data))
    # end for
    ########################################################################
    ## normalized by output values
    r_val = np.concatenate(r_ext_val, axis=0).astype(np.float) / l2_val
    l_val = np.concatenate(l_ext_val, axis=0).astype(np.float) / l2_val
    ########################################################################
    ## node count
    rl_diff = (t_img - n_img).reshape((-1,))
    ########################################################################
    df_val = np.linalg.norm(rl_diff)
    if (normalize == False):
        df_val = 1.0
    # end if
    ########################################################################
    # split concat data
    ########################################################################
    p_len = []
    x_key = []
    for key, arr in x_spt.items():
        x_key.append(key)
        p_len.append(arr.shape[0])
    # end for
    line = np.asarray(x_key).argsort()
    q_len = [p_len[j] for j in line] 
    x_len = []
    total_len = 0
    for val in q_len:
        total_len += val
        x_len.append(total_len)
    # end for
    ########################################################################
    new_bool = ro_new
    x_bool = np.split(new_bool, x_len, axis=0)[:-1]
    ########################################################################
    l_split  = np.split(l_val, x_len, axis=0)[:-1]
    r_split  = np.split(r_val, x_len, axis=0)[:-1]
    ########################################################################
    trans_dict = {}
    l_dict = {}
    r_dict = {}
    x_spt_keys = list(x_spt)
    for key in x_spt_keys:
        bool0 = x_bool[key][:-1]
        bool1 = x_bool[key][1:]
        trans_dict[key] = (np.sum((bool0 != bool1), axis=1)).tolist()
        l_dict[key]  = l_split[key]
        r_dict[key]  = r_split[key]
    # end for
    ########################################################################
    # get extended gradient data
    ########################################################################
    hess_list = []
    ########################################################################
    for j, trans_list in trans_dict.items():
        ####################################################################
        for did, n_trans in enumerate(trans_list):
            ################################################################
            # if (n_trans == 1):
            if (n_trans > 0):
                ############################################################
                dv_0 = (l_dict[j][did] - r_dict[j][did]) / df_val
                dv_1 = (l_dict[j][did + 1] - r_dict[j][did + 1]) / df_val
                ############################################################
                hess_list.append(dv_1 - dv_0)
                ############################################################
            # end if
            ################################################################
        # end for
        ####################################################################
    # end for
    ########################################################################
    hess_arr = np.array(hess_list, dtype=np.float).reshape((-1, num_classes))
    # hess_arr = np.array(hess_list, dtype=np.float)
    # print('div:', num_sample, ', len:', hess_arr.shape[0])
    ########################################################################
    return hess_arr, df_val
#############################################################################
#############################################################################
def extend_grad(model, train_ds, t_idx, n_idx, l2_out,
                x_site, x_div, normalize=True):
    #########################################################################
    x_spt = search_div(model, train_ds, t_idx, n_idx, l2_out, x_div)
    #########################################################################
    # get relu profile and nn_val at nodes
    #########################################################################
    p_loc = []
    x_key = []
    for key, arr in x_spt.items():
        x_key.append(key)
        p_loc.append(arr)
    # end for
    line = np.asarray(x_key).argsort()
    q_loc = [p_loc[j] for j in line]
    x_loc = np.concatenate(q_loc).reshape((-1,))
    #########################################################################
    x_val = []
    l2_val = []
    t_img = train_ds[t_idx][0]
    n_img = train_ds[n_idx][0]
    for t in x_loc:
        x_val.append((1 - t) * t_img + t * n_img)
        l2_val.append((1 - t) * l2_out[t_idx] + t * l2_out[n_idx])
    # end for
    l2_val = np.asarray(l2_val).reshape((-1, 1))
    x_new = np.asarray(x_val, dtype=np.float32)
    num_sample = x_new.shape[0]
    # NN's value
    nn = lambda x: model.get_act(x)
    #########################################################################
    ## check ReLU on off
    ro_new = []
    out_val = []
    ro_cut = {}
    for i in range(0, num_sample, test_bat):
        ####################################################################
        xx = chainer.Variable(xp.asarray(
            (x_new[i:i + test_bat]).astype(np.float32)))
        ####################################################################
        with chainer.using_config('train', False):
            ro_data = nn(xx)
        # end with
        ####################################################################
        ro_cut[i] = ro_data
        ####################################################################
        ro_arr = [val.reshape(val.shape[0], -1) for val in ro_data]
        ro_new.append(cpu(xp.concatenate(ro_arr, axis=1)))
        ####################################################################
        out_val.append(model.output)
        ####################################################################
    # end for
    ########################################################################
    ro_new = np.concatenate(ro_new, axis=0)
    ########################################################################
    ## normalized by output values
    nn_val = np.concatenate(out_val, axis=0).astype(np.float) / l2_val
    ########################################################################
    # get extend nn_val
    # right side input value
    r_ext = [ t_img ] * num_sample
    r_ext = np.asarray(r_ext, dtype=np.float32)
    # right side input value
    l_ext = [ n_img ] * num_sample
    l_ext = np.asarray(l_ext, dtype=np.float32)
    # NN's value
    nn = lambda x, cut : model.cutoff(x, cut)
    # get nn extended activation
    r_ext_val = []
    l_ext_val = []
    ########################################################################
    for i in range(0, num_sample, test_bat):
        ####################################################################
        rr = chainer.Variable(xp.asarray(
            (r_ext[i:i + test_bat]).astype(np.float32)))
        ll = chainer.Variable(xp.asarray(
            (l_ext[i:i + test_bat]).astype(np.float32)))
        ####################################################################
        cut_arr = ro_cut[i]
        ####################################################################
        with chainer.using_config('train', False):
            r_nn = nn(rr, cut_arr)
            l_nn = nn(ll, cut_arr)
        # end with
        r_ext_val.append(cpu(r_nn.data))
        l_ext_val.append(cpu(l_nn.data))
    # end for
    ########################################################################
    ## normalized by output values
    r_val = np.concatenate(r_ext_val, axis=0).astype(np.float) / l2_val
    l_val = np.concatenate(l_ext_val, axis=0).astype(np.float) / l2_val
    ########################################################################
    ## node count
    rl_diff = (t_img - n_img).reshape((-1,))
    ########################################################################
    df_val = np.linalg.norm(rl_diff)
    if (normalize == False):
        df_val = 1.0
    # end if
    ########################################################################
    # split concat data
    ########################################################################
    p_len = []
    x_key = []
    for key, arr in x_spt.items():
        x_key.append(key)
        p_len.append(arr.shape[0])
    # end for
    line = np.asarray(x_key).argsort()
    q_len = [p_len[j] for j in line] 
    x_len = []
    total_len = 0
    for val in q_len:
        total_len += val
        x_len.append(total_len)
    # end for
    #########################################################################
    new_bool = ro_new
    x_bool = np.split(new_bool, x_len, axis=0)[:-1]
    #########################################################################
    nn_split = np.split(nn_val, x_len, axis=0)[:-1]
    l_split  = np.split(l_val, x_len, axis=0)[:-1]
    r_split  = np.split(r_val, x_len, axis=0)[:-1]
    #########################################################################
    trans_dict = {}
    nn_dict = {}
    l_dict = {}
    r_dict = {}
    x_spt_keys = list(x_spt)
    for key in x_spt_keys:
        bool0 = x_bool[key][:-1]
        bool1 = x_bool[key][1:]
        trans_dict[key] = (np.sum((bool0 != bool1), axis=1)).tolist()
        nn_dict[key] = nn_split[key]
        l_dict[key]  = l_split[key]
        r_dict[key]  = r_split[key]
    # end for
    #########################################################################
    # get extended gradient data
    #########################################################################
    d_data = {}
    d_grad = {}
    d_node = {}
    gap_warn  = 0
    slip_warn = 0
    #########################################################################
    for j, trans_list in trans_dict.items():
        ####################################################################
        data_list = []
        grad_list = []
        node_set = set([])
        cid = 0
        ####################################################################
        for did, n_trans in enumerate(trans_list):
            ################################################################
            if (n_trans == 1):
                #############################################################
                if (did > cid):
                    #########################################################
                    dv_0 = (l_dict[j][did] - r_dict[j][cid]) / df_val
                    #########################################################
                else:
                    #########################################################
                    # print('Manual Runtime Warning: Slip', cid, did)
                    slip_warn += 1
                    eid = cid - 1
                    dv_0 = (l_dict[j][did] - r_dict[j][eid]) / df_val
                    #########################################################
                # end if
                #############################################################
                if ( (x_site[j][0] < x_spt[j][did]) and
                     (x_spt[j][did+1] < x_site[j][1]) ):
                    data_list.append(
                        (nn_dict[j][did] + nn_dict[j][did+1]) / 2.0)
                    grad_list.append(dv_0)
                    node_set |= set([(x_spt[j][did] + x_spt[j][did+1]) / 2.0])
                # end if
                #############################################################
                cid = did + 1
                #############################################################
            elif (n_trans > 1):
                #############################################################
                # print('Manual Runtime Warning: Large Gap:', n_trans, did)
                gap_warn += 1
                #############################################################
                if (did > cid):
                    #########################################################
                    dv_0 = (l_dict[j][did] - r_dict[j][cid]) / df_val
                    #########################################################
                else:
                    #########################################################
                    # print('Manual Runtime Warning: Slip Error', cid, did)
                    slip_warn += 1
                    eid = cid - 1
                    dv_0 = (l_dict[j][did] - r_dict[j][eid]) / df_val
                    #########################################################
                # end if
                #############################################################
                if ( (x_site[j][0] < x_spt[j][did]) and
                     (x_spt[j][did+1] < x_site[j][1]) ):
                    data_list.append(
                        (nn_dict[j][did] + nn_dict[j][did+1]) / 2.0)
                    grad_list.append(dv_0)
                    node_set |= set([(x_spt[j][did] + x_spt[j][did+1]) / 2.0])
                # end if
                #############################################################
                cid = did + 1
                #############################################################
            # end if
        # end for
        #####################################################################
        d_data[j] = np.array(data_list, dtype=np.float)
        d_grad[j] = np.array(grad_list, dtype=np.float)
        #####################################################################
        node_list = list(node_set)
        node_list = sorted(node_list)
        d_node[j] = np.array(node_list, dtype=np.float)
        #####################################################################
    # end for
    #########################################################################
    #########################################################################
    return d_data, d_grad, d_node, df_val
#############################################################################
#############################################################################
def calc_grad(model, train_ds, t_idx, n_idx):
    #########################################################################
    x_info = np.linspace(0 - 0.01, 1 + 0.01, 1020, dtype=np.float)
    #########################################################################
    min_val = 1e-05
    #########################################################################
    vr0 = 0.0
    vr1 = 1.0
    n_flag = False
    c_num = 0
    # print('start calculation for gradient')
    while(n_flag == False):
        n_flag, flag_sign, x_info, gd_cnt, gd_lst, gd_idx, gd_pos = calc_node(
            model, train_ds, t_idx, n_idx, x_info)
        c_num += 1
        min_node = (x_info[1:] - x_info[:-1]).min()
        # print('min node: {}'.format(min_node))
        if( min_node < sys.float_info.epsilon ):
            n_flag = True
    #########################################################################
    x_val = []
    for t in x_info:
        x_val.append(
            (1 - t) * train_ds[t_idx][0] + t * train_ds[n_idx][0])
    # end for
    x_new = np.asarray(x_val, dtype=np.float32)
    num_sample = x_new.shape[0]
    # NN's value
    nn = lambda x: model.get_act(x)
    ## check ReLU on off
    ro_new = []
    out_val = []
    for i in range(0, num_sample, test_bat):
        xx = chainer.Variable(xp.asarray(
            (x_new[i:i + test_bat]).astype(np.float32)))
        with chainer.using_config('train', False):
            ro_data = nn(xx)
        # end with
        ro_arr = [val.reshape(val.shape[0], -1) for val in ro_data]
        ro_new.append(cpu(xp.concatenate(ro_arr, axis=1)))
        out_val.append(model.output)
    # end for
    ########################################################################
    ro_new = np.concatenate(ro_new, axis=0)
    nn_val = cpu(xp.concatenate(out_val, axis=0))
    new_bool = ro_new
    bool0 = new_bool[:-1]
    bool1 = new_bool[1:]
    trans_list = (np.sum((bool0 != bool1), axis=1)).tolist()
    xx_new = x_new.reshape((num_sample, -1))
    ## node count
    data_list = []
    grad_list = []
    node_set = set([])
    cid = 0
    gap_warn = 0
    slip_warn = 0
    for did, n_trans in enumerate(trans_list):
        if (n_trans == 1):
            #################################################################
            if (did > cid):
                #############################################################
                df_val = np.linalg.norm(xx_new[did]-xx_new[cid])
                dv_0 = (nn_val[did]-nn_val[cid]) / df_val
                #############################################################
            else:
                #############################################################
                # print('Manual Runtime Warning: Slip', cid, xx_new[cid])
                slip_warn += 1
                eid = cid - 1
                df_val = np.linalg.norm(xx_new[did]-xx_new[eid])
                dv_0 = (nn_val[did]-nn_val[eid]) / df_val
                #############################################################
            # end if
            #################################################################
            if ( (vr0 < x_info[did]) and (x_info[did+1] < vr1) ):
                if (df_val > min_val):
                    data_list.append((nn_val[did] + nn_val[did+1]) / 2.0)
                    grad_list.append(dv_0)
                    node_set |= set([(x_info[did] + x_info[did+1]) / 2.0])
                # end if
            # end if
            #################################################################
            cid = did + 1
            #################################################################
        elif (n_trans > 1):
            #################################################################
            # print('Manual Runtime Warning: Large Gap:', n_trans, did)
            gap_warn += 1
            #################################################################
            if (did > cid):
                #############################################################
                df_val = np.linalg.norm(xx_new[did]-xx_new[cid])
                dv_0 = (nn_val[did]-nn_val[cid]) / df_val
                #############################################################
            else:
                #############################################################
                # print('Manual Runtime Warning: Error', cid, did)
                slip_warn += 1
                eid = cid - 1
                df_val = np.linalg.norm(xx_new[did]-xx_new[eid])
                dv_0 = (nn_val[did]-nn_val[eid]) / df_val
                #############################################################
            # end if
            #################################################################
            if ( (vr0 < x_info[did]) and (x_info[did+1] < vr1) ):
                if (df_val > min_val):
                    data_list.append((nn_val[did] + nn_val[did+1]) / 2.0)
                    grad_list.append(dv_0)
                    node_set |= set([(x_info[did] + x_info[did+1]) / 2.0])
                # end if
            # end if
            #################################################################
            cid = did + 1
            #################################################################
        # end if
    # end for
    #########################################################################
    data_arr = np.array(data_list, dtype=np.float)
    grad_arr = np.array(grad_list, dtype=np.float)
    #########################################################################
    node_list = list(node_set)
    node_list = sorted(node_list)
    node_arr = np.array(node_list, dtype=np.float)
    #########################################################################
    """
    #########################################################################
    hess_arr = grad_arr[1:] - grad_arr[:-1]
    node_val = node_arr.shape[0]
    grad_std  = grad_arr.std()
    hess_mean = hess_arr.mean()
    hess_std  = hess_arr.std()
    diffusion = hess_std * np.sqrt(node_val)
    inter_node = node_arr[1:] - node_arr[:-1]
    i_node_mean = inter_node.mean()
    i_node_std = inter_node.std()
    #########################################################################
    """
    #########################################################################
    return c_num, gap_warn, slip_warn, data_arr, grad_arr, node_arr
#############################################################################
"""
#############################################################################
def grad_output(model, ideal, p_train, nunits, idx):
    @training.make_extension(trigger=(num_grad, 'epoch'))
    def _grad_output(trainer):
        global ep_grad
        global node_value
        global gap_value
        c_num, g_warn, s_warn, grad_arr, node_arr = calc_grad(model)
        hess_arr = grad_arr[1:] - grad_arr[:-1]
        node_val = node_arr.shape[0]
        gap_value[nunits].append(g_warn)
        node_value[nunits].append(node_val)
        grad_std  = grad_arr.std()
        hess_mean = hess_arr.mean()
        hess_std  = hess_arr.std()
        inter_node = node_arr[1:] - node_arr[:-1]
        #####################################################################
        x_val, _ = get_dataset(num_test_data, total_range)
        x_val = x_val.reshape((-1,1)).astype(np.float32)
        xx = chainer.Variable(xp.asarray(x_val))
        # NN's value
        nn = lambda x: cpu(model(x).data)
        # calc difference
        with chainer.using_config('train', False):
            y_val = nn(xx)
        i_val = ideal(x_val)
        d_diff = y_val - i_val
        d_norm = np.linalg.norm(d_diff)
        d_diffusion = hess_arr.std() * np.sqrt(node_val)
        i_node_mean = inter_node.mean()
        i_node_std = inter_node.std()
        #####################################################################
        # plot
        plt.cla()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_val, y_val, label="nn")
        ax.plot(x_val, i_val, label="ideal")
        ax.plot(p_train['x'], p_train['y'], '.', label="sample dots")
        ax.legend()
        ax.set_title(
            'units: {}, epoch: {}, node: {}, '.format(
                nunits, ep_grad, node_val) +
            'norm: {:.05f}, diffusion {:.05f}\n'.format(d_norm, d_diffusion) +
            'grad std: {:.05f}, hess mean: {:.05f}, hess std: {:.05f}\n'.format(
                grad_std, hess_mean, hess_std) +
            'inter node mean: {:.05f}, inter node std: {:.05f}\n'.format(
                i_node_mean, i_node_std) + 
            'cycle: {}, gap: {}, slip: {}'.format(c_num, g_warn, s_warn),
            fontsize=9)
        plt.tight_layout()
        plt.savefig('result/U{:04d}_id{:03d}_ep{:03d}_plot.png'.format(
            nunits, idx, ep_grad))
        ax.clear()
        fig.clf()
        plt.close(fig)
        #####################################################################
        # grad plot
        plt.cla()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(node_arr, grad_arr, label="gradient")
        ax.plot(node_arr[:-1], inter_node, label="inter node")
        ax.set_title(
            'units: {}, epoch: {}, node: {}, '.format(
                nunits, ep_grad, node_val) +
            'norm: {:.05f}, diffusion {:.05f}\n'.format(d_norm, d_diffusion) +
            'grad std: {:.05f}, hess mean: {:.05f}, hess std: {:.05f}\n'.format(
                grad_std, hess_mean, hess_std) +
            'inter node mean: {:.05f}, inter node std: {:.05f}\n'.format(
                i_node_mean, i_node_std) + 
            'cycle: {}, gap: {}, slip: {}'.format(c_num, g_warn, s_warn),
            fontsize=9)
        plt.tight_layout()
        ax.legend()
        plt.savefig('result/U{:04d}_id{:03d}_ep{:03d}_grad.png'.format(
            nunits, idx, ep_grad))
        ax.clear()
        fig.clf()
        plt.close(fig)
        #####################################################################
        # hessian plot
        plt.cla()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(node_arr[1:], hess_arr, label="hessian")
        ax.legend()
        ax.set_title(
            'units: {}, epoch: {}, node: {}, '.format(
                nunits, ep_grad, node_val) +
            'norm: {:.05f}, diffusion {:.05f}\n'.format(d_norm, d_diffusion) +
            'grad std: {:.05f}, hess mean: {:.05f}, hess std: {:.05f}\n'.format(
                grad_std, hess_mean, hess_std) +
            'inter node mean: {:.05f}, inter node std: {:.05f}\n'.format(
                i_node_mean, i_node_std) + 
            'cycle: {}, gap: {}, slip: {}'.format(c_num, g_warn, s_warn),
            fontsize=9)
        plt.tight_layout()
        plt.savefig('result/U{:04d}_id{:03d}_ep{:03d}_hess.png'.format(
            nunits, idx, ep_grad))
        ax.clear()
        fig.clf()
        plt.close(fig)
        #####################################################################
        # inter node plot
        plt.cla()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(node_arr[:-1], inter_node, label="inter node")
        ax.set_title(
            'units: {}, epoch: {}, node: {}, '.format(
                nunits, ep_grad, node_val) +
            'norm: {:.05f}, diffusion {:.05f}\n'.format(d_norm, d_diffusion) +
            'grad std: {:.05f}, hess mean: {:.05f}, hess std: {:.05f}\n'.format(
                grad_std, hess_mean, hess_std) +
            'inter node mean: {:.05f}, inter node std: {:.05f}\n'.format(
                i_node_mean, i_node_std) + 
            'cycle: {}, gap: {}, slip: {}'.format(c_num, g_warn, s_warn),
            fontsize=9)
        plt.tight_layout()
        ax.legend()
        plt.savefig('result/U{:04d}_id{:03d}_ep{:03d}_inode.png'.format(
            nunits, idx, ep_grad))
        ax.clear()
        fig.clf()
        plt.close(fig)
        #####################################################################
        ep_grad += num_grad
    return _grad_output
#############################################################################
"""
#############################################################################
