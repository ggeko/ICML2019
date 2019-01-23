import numpy as np
import chainer
import sys
import os

from PIL import Image

from chainer.backends import cuda


xp = cuda.cupy
cpu = cuda.to_cpu

############################################################################
############################################################################
def get_dataset(ds_name, model_name, data_dir, ds_size=None,
                ds_normalized=True, ds_amount=50000):
    ########################################################################
    if ds_name == 'MNIST':
        ####################################################################
        if model_name == 'MLP':
            ndim = 1
        else:
            ndim = 3
        # end if
        train_ds, test_ds = chainer.datasets.get_mnist(ndim=ndim)
        ####################################################################
    elif ds_name == 'CIFAR10':
        ####################################################################
        if model_name == 'MLP':
            ndim = 1
        else:
            ndim = 3
        # end if
        train_ds, test_ds = chainer.datasets.get_cifar10(ndim=ndim)
        ####################################################################
    elif ds_name == 'CIFAR100':
        ####################################################################
        if model_name == 'MLP':
            ndim = 1
        else:
            ndim = 3
        # end if
        train_ds, test_ds = chainer.datasets.get_cifar100(ndim=ndim)
        ####################################################################
    # end if
    ########################################################################
    if ((ds_size is not None) or (ds_normalized == True) or
        (ds_amount is not None)):
        ####################################################################
        train_ds, test_ds = remake_dataset(
            train_ds, test_ds, ds_name, ds_size, ndim,
            data_dir, ds_normalized, ds_amount)
        ####################################################################
    # end if
    return train_ds, test_ds
############################################################################
############################################################################
def image_trans(x_data, ds_name, ds_size):
    x_data = (x_data * 255.0).astype(np.uint8)
    if (ds_name == 'CIFAR10') or (ds_name == 'CIFAR100'):
        x_data = x_data.transpose((0, 2, 3, 1))
    # end if
    images = []
    s_size = (ds_size, ds_size)
    for org_img in x_data:
        p_img = Image.fromarray(np.uint8(org_img))
        ## s_img = p_img.resize(s_size, Image.BILINEAR)
        ## s_img = p_img.resize(s_size, Image.BICUBIC)
        s_img = p_img.resize(s_size, Image.LANCZOS)
        np_img = np.array(s_img, dtype=np.uint8)
        images.append(np.array(s_img, dtype=np.uint8))
    # end for
    x_data = np.asarray(images, dtype=np.float32) / 255.0
    if (ds_name == 'CIFAR10') or (ds_name == 'CIFAR100'):
        x_data = x_data.transpose((0, 3, 1, 2))
    # end if
    return x_data
############################################################################
############################################################################
def calc_standard(idata):
    # calc standardization data
    nn = idata.shape[0]
    data_shape = idata.shape
    idata = (idata.reshape( (nn, -1) )).copy()
    l_mean = idata.mean(axis=0)
    idata -= l_mean
    l_std = idata.std(axis=0)
    return l_mean, l_std
############################################################################
############################################################################
def sdd_by_data(idata, l_mean, l_std):
    # Standardization (Local Contrast Normalization) 
    nn = idata.shape[0]
    data_shape = idata.shape
    idata = idata.reshape( (nn, -1) )
    idata -= l_mean
    idata /= (l_std + 1e-5)
    return idata.reshape( data_shape )
############################################################################
############################################################################
def L2normalize(idata):
    nn = idata.shape[0]
    data_shape = idata.shape
    idata = idata.reshape( (nn, -1) )
    arr = xp.asarray(np.asarray(idata, dtype=np.float32))
    l2_arr = xp.sqrt((arr * arr).sum(axis=1)).reshape((-1,1))
    # l2_mean = l2_arr.mean()
    l2_mean = np.sqrt(idata.shape[1])
    idata = cpu(arr * l2_mean / l2_arr)
    # idata = cpu(arr / l2_arr)
    return idata.reshape( data_shape )
############################################################################
############################################################################
def data_normalize(x_train, x_test):
    ########################################################################
    train_mean, train_std = calc_standard(x_train)
    x_train = sdd_by_data(x_train, train_mean, train_std)
    x_test  = sdd_by_data(x_test,  train_mean, train_std)
    ########################################################################
    # L2 normalized
    ########################################################################
    x_train = L2normalize(x_train)
    x_test  = L2normalize(x_test)
    ########################################################################
    return x_train, x_test
############################################################################
############################################################################
def remake_dataset(train_ds, test_ds, ds_name, ds_size, ndim, data_dir,
                   ds_normalized=True, ds_amount=50000):
    ########################################################################
    # train data
    x_train = []
    t_train = []
    for lid in range(len(train_ds)):
        x_train.append(train_ds[lid][0])
        t_train.append(train_ds[lid][1])
    # end for
    t_train = np.asarray(t_train, dtype=np.int32)[:ds_amount]
    x_train = np.asarray(x_train, dtype=np.float32)[:ds_amount]
    ########################################################################
    # test data
    x_test = []
    t_test = []
    for lid in range(len(test_ds)):
        x_test.append(test_ds[lid][0])
        t_test.append(test_ds[lid][1])
    # end for
    t_test = np.asarray(t_test, dtype=np.int32)[:ds_amount]
    x_test = np.asarray(x_test, dtype=np.float32)[:ds_amount]
    ########################################################################
    if (ds_normalized == True):
        core_name ='_{}_s{}_d{}_a{}_N.npy'.format(
            ds_name, ds_size, ndim, ds_amount)
    else:
        core_name ='_{}_s{}_d{}_a{}.npy'.format(
            ds_name, ds_size, ndim, ds_amount)
    # end if
    ########################################################################
    train_name = data_dir + '/x_train' + core_name
    test_name  = data_dir + '/x_test' + core_name
    if (os.path.isfile(train_name) and os.path.isfile(test_name)):
        x_train = np.load(train_name)
        x_test  = np.load(test_name)
    else:
        ####################################################################
        train_sp = x_train.shape
        test_sp = x_test.shape
        ####################################################################
        if (ds_name == 'MNIST'):
            ch_size = 1
            fig_shape = [28, 28]
        elif (ds_name == 'CIFAR10') or (ds_name == 'CIFAR100'):
            ch_size = 3
            fig_shape = [3, 32, 32]
        # end if
        x_train = x_train.reshape([train_sp[0]] + fig_shape)
        x_test  = x_test.reshape([test_sp[0]] + fig_shape)
        ####################################################################
        x_train = image_trans(x_train, ds_name, ds_size)
        x_test  = image_trans(x_test, ds_name, ds_size)
        ####################################################################
        if (ndim == 3):
            train_shape = (train_sp[0], train_sp[1], ds_size, ds_size)
            test_shape = (test_sp[0], test_sp[1], ds_size, ds_size)
        elif (ndim == 1):
            train_shape = (train_sp[0], ch_size * ds_size * ds_size)
            test_shape = (test_sp[0], ch_size * ds_size * ds_size)
        # end if
        x_train = x_train.reshape(train_shape)
        x_test  = x_test.reshape(test_shape)
        ####################################################################
        if (ds_normalized == True):
            x_train, x_test = data_normalize(x_train, x_test)
        # end if
        ####################################################################
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        # end if
        np.save(train_name, x_train)
        np.save(test_name, x_test)
        ####################################################################
    # end if
    ########################################################################
    train_ds = chainer.datasets.TupleDataset(x_train, t_train)
    test_ds  = chainer.datasets.TupleDataset(x_test, t_test)
    ########################################################################
    return train_ds, test_ds
############################################################################
############################################################################
############################################################################
def setup_data(train_ds, test_ds, num_rl, data_dir):
    ########################################################################
    # train data
    x_train = []
    t_train = []
    for lid in range(len(train_ds)):
        x_train.append(train_ds[lid][0])
        t_train.append(train_ds[lid][1])
    # end for
    t_train = np.asarray(t_train, dtype=np.int32)
    x_train = np.asarray(x_train, dtype=np.float32)
    ########################################################################
    # test data
    x_test = []
    t_test = []
    for lid in range(len(test_ds)):
        x_test.append(test_ds[lid][0])
        t_test.append(test_ds[lid][1])
    # end for
    t_test = np.asarray(t_test, dtype=np.int32)
    x_test = np.asarray(x_test, dtype=np.float32)
    ########################################################################
    # random label
    file_name = data_dir + '/t_rand_{}.npy'.format(num_rl)
    if os.path.isfile(file_name):
        t_rand = np.load(file_name)
    else:
        t_train_rand = t_train[:num_rl]
        t_train_true = t_train[num_rl:]
        perm = np.random.permutation(num_rl)
        t_rand_rand  = t_train_rand[perm]
        t_rand = np.concatenate([t_rand_rand, t_train_true], axis=0)
        t_rand = t_rand.astype(np.int32)
        ####################################################################
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        # end if
        np.save(data_dir + '/t_rand_{}.npy'.format(num_rl), t_rand)
        ####################################################################
    # end if
    rand_ds = chainer.datasets.TupleDataset(x_train, t_rand)
    ########################################################################
    return x_train, t_train, t_rand, rand_ds, x_test, t_test
############################################################################
############################################################################
def get_label_dict(t_data, data_dir):
    l_dict = {}
    num_class = t_data.max() + 1
    for i in range(num_class):
        l_dict[i] = []
    # end for
    for idx, lab in enumerate(t_data.tolist()):
        l_dict[lab].append(idx)
    # end for
    for i in range(num_class):
        l_dict[i] = np.asarray(l_dict[i])
    # end for
    return l_dict
############################################################################
############################################################################
def get_neib_class_arr(trgt_class_arr, num_class, pair_type):
    if(pair_type == 'same'):
        c_arr = trgt_class_arr
    elif (pair_type == 'diff'):
        trgt_num = trgt_class_arr.shape[0]
        shift_arr = np.random.randint(1, num_class, trgt_num)
        c_arr = []
        for t_class, s_val in zip(trgt_class_arr, shift_arr):
            n_class = (t_class + s_val) % num_class
            c_arr.append(n_class)
        # end for
        c_arr = np.asarray(c_arr, dtype=np.int)
    else:
        print('error: pair_type warning:', pair_type)
        sys.exit(0)
    # end if
    return c_arr
############################################################################
############################################################################
def make_pair_ids(trgt_num, t_data, data_dir,
                  pair_type='same', data_type='train', ds_amount=50000):
    ########################################################################
    pair_name = '/pair_ids_t{}_{}_{}_{}.npy'.format(
        trgt_num, pair_type, data_type, ds_amount)
    file_name = data_dir + pair_name
    ########################################################################
    if os.path.isfile(file_name):
        ####################################################################
        pair_ids = np.load(file_name)
        ####################################################################
    elif (pair_type == 'rand'):
        ####################################################################
        n_sample = t_data.shape[0]
        ran_index = np.random.permutation(n_sample)
        if (n_sample > trgt_num * 2):
            t_ids = ran_index[:trgt_num]
            n_ids = ran_index[trgt_num:2*trgt_num]
            # print('shape', t_ids.shape, n_ids.shape)
            pair_ids = np.asarray([t_ids, n_ids]).transpose((1, 0))
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            # end if
            np.save(file_name, pair_ids)
            # print('shape', pair_ids.shape)
        else:
            print('error: too large trgt_num.')
            sys.exit(0)
        # end if
        ####################################################################
    else:
        ####################################################################
        label_dict = get_label_dict(t_data, data_dir)
        num_class = t_data.max() + 1
        t_class_arr = np.random.randint(0, num_class, trgt_num)
        n_class_arr = get_neib_class_arr(t_class_arr, num_class, pair_type)
        amount_label = []
        for l_arr in label_dict.values():
            amount_label.append(l_arr.shape[0])
        # end for
        min_amount = min(amount_label)
        t_select = np.random.randint(0, min_amount, trgt_num)
        n_select = np.random.randint(0, min_amount, trgt_num)
        pair_ids = []
        for t_c, n_c, t_id, n_id in zip(
                t_class_arr, n_class_arr, t_select, n_select):
            if (t_c == n_c) and (t_id == n_id):
                ran_shift = np.random.randint(1, min_amount)
                n_id = (n_id + ran_shift) % min_amount
            # end if
            t_idx = label_dict[t_c][t_id]
            n_idx = label_dict[n_c][n_id]
            pair_ids.append([t_idx, n_idx])
        # end for
        pair_ids = np.asarray(pair_ids)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        # end if
        np.save(file_name, pair_ids)
    # end if
    ########################################################################
    return pair_ids
############################################################################
############################################################################
def get_near_pair_index(trgt_num, model, x_train, t_train, t_rand, num_rl,
                        data_dir, ds_size=None, def_neibor='input',
                        same_label=True):
    ########################################################################
    if (ds_size is None):
        return get_pair_index(trgt_num, model, x_train, t_train,
                              t_rand, num_rl, data_dir, def_neibor)
    else:
        return get_pair_index_resize(
            trgt_num, model, x_train, t_train,
            t_rand, num_rl, ds_size, data_dir, def_neibor, same_label)
    # end if
    ########################################################################
############################################################################
############################################################################
def get_pair_index(trgt_num, model, x_train, t_train, t_rand,
                   num_rl, data_dir, def_neibor='input'):
    ########################################################################
    file_name = data_dir + '/trgt_data_rl{}_t{}.npy'.format(num_rl, trgt_num)
    if os.path.isfile(file_name):
        trgt_data = np.load(file_name)
    else:
        trgt_data = set_target_data(
            t_train, t_rand, num_rl, trgt_num, data_dir)
    # end if
    ########################################################################
    trgt_ids = []
    for (tid, _, _) in trgt_data:
        trgt_ids.append(tid)
    # end for
    ########################################################################
    file_name = data_dir + '/pair_ids_t{}.npy'.format(trgt_num)
    if os.path.isfile(file_name):
        ####################################################################
        pair_ids = np.load(file_name)
        ####################################################################
    else:
        ####################################################################
        if (def_neibor == 'input'):
            ################################################################
            file_name = data_dir + '/input_lengths.npy'
            if os.path.isfile(file_name):
                lengths = np.load(file_name)
            else:
                lengths = calc_input_distance(x_train, data_dir)
            # end if
            ################################################################
        elif (def_neibor == 'output'):
            ################################################################
            file_name = data_dir + '/output_lengths.npy'
            if os.path.isfile(file_name):
                lengths = np.load(file_name)
            else:
                lengths = calc_output_distance(model, x_train, data_dir)
            # end if
            ################################################################
        else:
            print('error: def_neibor')
            sys.exit()
        # end if
        ####################################################################
        pair_ids = []
        ####################################################################
        for target_idx in trgt_ids:
            ################################################################
            nei_idx = np.argsort(lengths[target_idx], axis=0)[1]
            ################################################################
            pair_ids.append( [target_idx, nei_idx] )
            ################################################################
        # end for
        ####################################################################
        pair_ids = np.array(pair_ids)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        # end if
        file_name = data_dir + '/pair_ids_t{}.npy'.format(trgt_num)
        np.save(file_name, pair_ids)
        ####################################################################
    # end if
    ########################################################################
    return pair_ids, trgt_data
############################################################################
############################################################################
def get_pair_index_resize(trgt_num, model, x_train, t_train, t_rand,
                          num_rl, ds_size, data_dir, def_neibor='input',
                          same_label = True):
    ########################################################################
    trgt_name = '/trgt_data_rl{}_t{}_s{}.npy'.format(num_rl, trgt_num, ds_size)
    file_name = data_dir + trgt_name
    if os.path.isfile(file_name):
        trgt_data = np.load(file_name)
    else:
        trgt_data = set_target_data(
            t_train, t_rand, num_rl, trgt_num, data_dir, ds_size)
    # end if
    ########################################################################
    trgt_ids = []
    for (tid, _, _) in trgt_data:
        trgt_ids.append(tid)
    # end for
    ########################################################################
    file_name = data_dir + '/pair_ids_t{}_s{}.npy'.format(trgt_num, ds_size)
    if os.path.isfile(file_name):
        ####################################################################
        pair_ids = np.load(file_name)
        ####################################################################
    else:
        ####################################################################
        if (def_neibor == 'input'):
            ################################################################
            file_name = data_dir + '/input_lengths_s{}.npy'.format(ds_size)
            if os.path.isfile(file_name):
                lengths = np.load(file_name)
            else:
                lengths = calc_input_distance(x_train, data_dir, ds_size)
            # end if
            ################################################################
        elif (def_neibor == 'output'):
            ################################################################
            file_name = data_dir + '/output_lengths.npy'
            if os.path.isfile(file_name):
                lengths = np.load(file_name)
            else:
                lengths = calc_output_distance(model, x_train, data_dir)
            # end if
            ################################################################
        else:
            print('error: def_neibor')
            sys.exit()
        # end if
        ####################################################################
        pair_ids = []
        ####################################################################
        for target_idx in trgt_ids:
            ################################################################
            nei_idx = np.argsort(lengths[target_idx], axis=0)[1]
            ################################################################
            pair_ids.append( [target_idx, nei_idx] )
            ################################################################
        # end for
        ####################################################################
        pair_ids = np.array(pair_ids)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        # end if
        pair_name = '/pair_ids_t{}_s{}.npy'.format(trgt_num, ds_size)
        file_name = data_dir + pair_name
        np.save(file_name, pair_ids)
        ####################################################################
    # end if
    ########################################################################
    return pair_ids, trgt_data
############################################################################
############################################################################
def set_target_data(t_train, t_rand, num_rl, trgt_num, data_dir, ds_size=None):
    ########################################################################
    t_slct_rand = np.random.randint(0, t_train.shape[0], trgt_num)
    trgt_data = []
    for i in range(trgt_num):
        kari_idx = t_slct_rand[i]
        trgt_lbl_true = t_train[kari_idx]
        trgt_lbl_rand = t_rand[kari_idx]
        trgt_data.append( [kari_idx, trgt_lbl_true, trgt_lbl_rand] )
    # end for
    ########################################################################
    trgt_data = np.array(trgt_data).reshape((-1,3))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # end if
    trgt_name = '/trgt_data_rl{}_t{}'.format(num_rl, trgt_num)
    if (ds_size is not None):
        trgt_name += '_s{}'.format(ds_size)
    # end if
    np.save(data_dir + trgt_name + '.npy', trgt_data)
    ########################################################################
    return trgt_data
############################################################################
############################################################################
def calc_output_distance(model, x_train, data_dir):
    ########################################################################
    # for output distance
    ########################################################################
    x_val = chainer.Variable(np.asarray(x_train))
    with chainer.using_config('train', False):
        activations = model(x_val).data
    # end with
    arr = xp.asarray(np.asarray(activations, dtype=np.float32))
    n_elem = arr.shape[0]
    xmat = []
    for i in range(n_elem):
        sub_arr = arr[i:i+1]
        seq = cpu(((arr-sub_arr)*(arr-sub_arr)).sum(axis=1))
        xmat.append(seq)
    # end for
    lengths = np.array(xmat, dtype=np.float32)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # end if
    np.save(data_dir + '/output_lengths.npy', lengths)
    ########################################################################
    return lengths
############################################################################
############################################################################
def calc_input_distance(x_train, data_dir, ds_size=None):
    ########################################################################
    # for input distance
    ########################################################################
    n_elem = x_train.shape[0]
    arr = x_train.reshape((n_elem, -1))
    arr = xp.asarray(np.asarray(arr, dtype=np.float32))
    xmat = []
    for i in range(n_elem):
        if (i%1000 == 0):
            print('step =', i)
        # end if
        sub_arr = arr[i:i+1]
        seq = cpu(((arr-sub_arr)*(arr-sub_arr)).sum(axis=1))
        xmat.append(seq)
    # end for
    lengths = np.array(xmat, dtype=np.float32)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # end if
    in_name = '/input_lengths'
    if (ds_size is not None):
        in_name += '_s{}'.format(ds_size)
    # end if
    np.save(data_dir + in_name + '.npy', lengths)
    ########################################################################
    return lengths
############################################################################
############################################################################
