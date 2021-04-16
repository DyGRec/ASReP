import sys
import copy
import random
import numpy as np
import multiprocessing
import time
import os
from collections import defaultdict


from metrics import *
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from tqdm import tqdm
import copy

Ks = [1, 2, 3, 4, 5, 10, 20, 40, 50, 60, 70, 80, 90,100]
cores = multiprocessing.cpu_count() // 2

def load_file_and_sort(filename, reverse=False, augdata=None, aug_num=0, M=10):
    data = defaultdict(list)
    max_uind = 0
    max_iind = 0
    with open(filename, 'r') as f:
        for line in f:
            one_interaction = line.rstrip().split("\t")
            uind = int(one_interaction[0]) + 1
            iind = int(one_interaction[1]) + 1
            max_uind = max(max_uind, uind)
            max_iind = max(max_iind, iind)
            t = float(one_interaction[2])
            data[uind].append((iind, t))
    print('data users: ', max_uind)
    print('data items: ', max_iind)
    print('data instances: ', sum([len(ilist) for _, ilist in data.items()]))
    if augdata:
        for u, ilist in augdata.items():
            sorted_interactions = sorted(ilist, key=lambda x:x[1])
            for i in range(min(aug_num, len(sorted_interactions))):
                if len(data[u]) >= M: continue
                data[u].append((sorted_interactions[i]))
        print('After augmentation:')
        print('data users: ', max_uind)
        print('data items: ', max_iind)
        print('data instances: ', sum([len(ilist) for user, ilist in data.items()]))

    sorted_data = {}
    for u, i_list in data.items():
        if not reverse:
            sorted_interactions = sorted(i_list, key=lambda x:x[1])
        else:
            sorted_interactions = sorted(i_list, key=lambda x:x[1], reverse=True)
        seq = [interaction[0] for interaction in sorted_interactions]
        sorted_data[u] = seq

    return sorted_data, max_uind, max_iind


def augdata_load(aug_filename):
    augdata = defaultdict(list)
    with open(aug_filename, 'r') as f:
        for line in f:
            one_interaction = line.rstrip().split("\t")
            uind = int(one_interaction[0]) + 1
            iind = int(one_interaction[1]) + 1
            t = float(one_interaction[2])
            augdata[uind].append((iind, t))

    return augdata



def data_load(data_name, args):
    reverseornot = args.reversed == 1
    if not reverseornot:
        train_file = f"./data/{data_name}/train.txt"
        valid_file = f"./data/{data_name}/valid.txt"
        test_file = f"./data/{data_name}/test.txt"
    else:
        train_file = f"./data/{data_name}/train_reverse.txt"
        valid_file = f"./data/{data_name}/valid_reverse.txt"
        test_file = f"./data/{data_name}/test_reverse.txt"

    original_train = None
    augdata = None
    if 'aug' in data_name or 'itemcor' in data_name:
        original_dataname = ''
        for substr in data_name.split('_')[:-1]:
            original_dataname += substr + '_'
        original_dataname = original_dataname[:-1]
        original_train_file = f"./data/{original_dataname}/train.txt"
        original_train, _, _ = load_file_and_sort(original_train_file)
    if args.aug_traindata > 0:
        original_train_file = f"./data/{data_name}/train.txt"
        original_train, _, _ = load_file_and_sort(original_train_file)
        aug_data_signature = './aug_data/{}/lr_{}_maxlen_{}_hsize_{}_nblocks_{}_drate_{}_l2_{}_nheads_{}_gen_num_'.format(args.dataset, args.lr, args.maxlen, args.hidden_units, args.num_blocks, args.dropout_rate, args.l2_emb, args.num_heads)
        gen_num_max = 20
        if os.path.exists(aug_data_signature + str(gen_num_max) + '_M_20.txt'):
            augdata = augdata_load(aug_data_signature + str(gen_num_max) + '_M_20.txt')
            print('load ', aug_data_signature + str(gen_num_max) + '_M_20.txt')
        else:
            gen_num_max = 10
            augdata = augdata_load(aug_data_signature + '10_M_20.txt')

    if args.aug_traindata > 0:
        user_train, train_usernum, train_itemnum = load_file_and_sort(train_file, reverse=reverseornot, augdata=augdata, aug_num=args.aug_traindata, M=args.M)
    else:
        user_train, train_usernum, train_itemnum = load_file_and_sort(train_file, reverse=reverseornot)
    user_valid, valid_usernum, valid_itemnum = load_file_and_sort(valid_file, reverse=reverseornot)
    user_test, test_usernum, test_itemnum = load_file_and_sort(test_file, reverse=reverseornot)

    usernum = max([train_usernum, valid_usernum, test_usernum])
    itemnum = max([train_itemnum, valid_itemnum, test_itemnum])

    print("num: ", len(user_valid), len(user_test), usernum, itemnum)

    return [user_train, user_valid, user_test, original_train, usernum, itemnum]


def data_augment(model, dataset, args, sess, gen_num):
    [train, valid, test, original_train, usernum, itemnum] = copy.deepcopy(dataset)
    all_users = list(train.keys())

    cumulative_preds = defaultdict(list)
    for num_ind in range(gen_num):
        batch_seq = []
        batch_u = []
        batch_item_idx = []

        for u_ind, u in enumerate(all_users):
            u_data = train.get(u, []) + valid.get(u, []) + test.get(u, []) + cumulative_preds.get(u, [])

            if len(u_data) == 0 or len(u_data) >= args.M: continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for i in reversed(u_data):
                if idx == -1: break
                seq[idx] = i
                idx -= 1
            rated = set(u_data)
            item_idx = list(set([i for i in range(itemnum)]) - rated)

            batch_seq.append(seq)
            batch_item_idx.append(item_idx)
            batch_u.append(u)

            if (u_ind + 1) % int(args.batch_size / 16) == 0 or u_ind + 1 == len(all_users):
                predictions = model.predict(sess, batch_u, batch_seq)
                for batch_ind in range(len(batch_item_idx)):
                    test_item_idx = batch_item_idx[batch_ind]
                    test_predictions = predictions[batch_ind][test_item_idx]
    
                    ranked_items_ind = list((-1*np.array(test_predictions)).argsort())
                    rankeditem_oneuserids = [int(test_item_idx[i]) for i in ranked_items_ind]

                    u_batch_ind = batch_u[batch_ind]
                    cumulative_preds[u_batch_ind].append(rankeditem_oneuserids[0])

                batch_seq = []
                batch_item_idx = []
                batch_u = []

    return cumulative_preds



def eval_one_interaction(x):
    results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }
    rankeditems = np.array(x[0])
    test_ind = x[1]
    scale_pred = x[2]
    test_item = x[3]
    r = np.zeros_like(rankeditems)
    r[rankeditems==test_ind] = 1
    if len(r) != len(scale_pred):
        r = rank_corrected(r, len(r)-1, len(scale_pred))
    gd_prob = np.zeros_like(rankeditems)
    gd_prob[test_ind] = 1

    for ind_k in range(len(Ks)):
        results["precision"][ind_k] += precision_at_k(r, Ks[ind_k])
        results["recall"][ind_k] += recall(rankeditems, [test_ind], Ks[ind_k])
        results["ndcg"][ind_k] += ndcg_at_k(r, Ks[ind_k], 1)
        results["hit_ratio"][ind_k] += hit_at_k(r, Ks[ind_k])
    results["auc"] += auc(gd_prob, scale_pred)
    results["mrr"] += mrr(r)

    return results


def rank_corrected(r, m, n):
    pos_ranks = np.argwhere(r==1)[:,0]
    corrected_r = np.zeros_like(r)
    for each_sample_rank in list(pos_ranks):
        corrected_rank = int(np.floor(((n-1)*each_sample_rank)/m))
        if corrected_rank >= len(corrected_r) - 1:
            continue
        corrected_r[corrected_rank] = 1
    assert np.sum(corrected_r) <= 1
    return corrected_r


def evaluate(model, dataset, args, sess, testorvalid):
    [train, valid, test, original_train, usernum, itemnum] = copy.deepcopy(dataset)
    results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }

    short_seq_results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }

    long_seq_results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }

    short7_seq_results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }

    short37_seq_results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }

    medium3_seq_results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }

    medium7_seq_results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }
    rs = []

    if testorvalid == "test":
        eval_data = test
    else:
        eval_data = valid
    num_valid_interactions = 0
    pool = multiprocessing.Pool(cores)

    all_predictions_results = []
    all_item_idx = []
    all_u = []

    batch_seq = []
    batch_u = []
    batch_item_idx = []


    u_ind = 0
    for u, i_list in eval_data.items():
        u_ind += 1
        if len(train[u]) < 1 or len(eval_data[u]) < 1: continue


        rated = set(train[u])
        rated.add(0)
        if testorvalid == "test":
            valid_set = set(valid.get(u, []))
            rated = rated | valid_set

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if testorvalid == "test":
            if u in valid:
                for i in reversed(valid[u]):
                    if idx == -1: break
                    seq[idx] = i
                    idx -= 1
        for i in reversed(train[u]):
            if idx == -1: break
            seq[idx] = i
            idx -= 1
        item_idx = [i_list[0]]
        if args.evalnegsample == -1:
            item_idx += list(set([i for i in range(1, itemnum+1)]) - rated - set([i_list[0]]))
        else:
            item_candiates = list(set([i for i in range(1, itemnum+1)]) - rated - set([i_list[0]]))
            if args.evalnegsample >= len(item_candiates):
                item_idx += item_candiates
            else:
                item_idx += list(np.random.choice(item_candiates, size=args.evalnegsample, replace=False))

        batch_seq.append(seq)
        batch_item_idx.append(item_idx)
        batch_u.append(u)

        if len(batch_u) % int(args.batch_size / 8) == 0 or u_ind == len(eval_data):
            predictions = model.predict(sess, batch_u, batch_seq)
            for pred_ind in range(predictions.shape[0]):
                all_predictions_results.append(predictions[pred_ind])
                all_item_idx.append(batch_item_idx[pred_ind])
                all_u.append(batch_u[pred_ind])

            batch_seq = []
            batch_item_idx = []
            batch_u = []


    rankeditems_list = []
    test_indices = []
    scale_pred_list = []
    test_allitems = []

    short_seq_rankeditems_list = []
    short_seq_test_indices = []
    short_seq_scale_pred_list = []
    short_seq_test_allitems = []

    short7_seq_rankeditems_list = []
    short7_seq_test_indices = []
    short7_seq_scale_pred_list = []
    short7_seq_test_allitems = []

    short37_seq_rankeditems_list = []
    short37_seq_test_indices = []
    short37_seq_scale_pred_list = []
    short37_seq_test_allitems = []

    medium3_seq_rankeditems_list = []
    medium3_seq_test_indices = []
    medium3_seq_scale_pred_list = []
    medium3_seq_test_allitems = []

    medium7_seq_rankeditems_list = []
    medium7_seq_test_indices = []
    medium7_seq_scale_pred_list = []
    medium7_seq_test_allitems = []


    long_seq_rankeditems_list = []
    long_seq_test_indices = []
    long_seq_scale_pred_list = []
    long_seq_test_allitems = []

    rankeditemid_list = []
    rankeditemid_scores = []

    all_predictions_results_output = []

    for ind in range(len(all_predictions_results)):
        test_item_idx = all_item_idx[ind]
        unk_predictions = all_predictions_results[ind][test_item_idx]

        scaler = MinMaxScaler()
        scale_pred = list(np.transpose(scaler.fit_transform(np.transpose(np.array([unk_predictions]))))[0])

        rankeditems_list.append(list((-1*np.array(unk_predictions)).argsort()))
        test_indices.append(0)
        test_allitems.append(test_item_idx[0])
        scale_pred_list.append(scale_pred)

        if 'aug' in args.dataset or 'itemco' in args.dataset or args.aug_traindata > 0:
            real_train = original_train
        else:
            real_train = train

        sorted_ind = list((-1*np.array(unk_predictions)).argsort())
        if len(real_train[all_u[ind]]) <= 3:
            short_seq_rankeditems_list.append(sorted_ind)
            short_seq_test_indices.append(0)
            short_seq_scale_pred_list.append(scale_pred)
            short_seq_test_allitems.append(test_item_idx[0])

        if len(real_train[all_u[ind]]) <= 7:
            short7_seq_rankeditems_list.append(sorted_ind)
            short7_seq_test_indices.append(0)
            short7_seq_scale_pred_list.append(scale_pred)
            short7_seq_test_allitems.append(test_item_idx[0])

        if len(real_train[all_u[ind]]) > 3 and len(real_train[all_u[ind]]) <= 7:
            short37_seq_rankeditems_list.append(sorted_ind)
            short37_seq_test_indices.append(0)
            short37_seq_scale_pred_list.append(scale_pred)
            short37_seq_test_allitems.append(test_item_idx[0])

        if len(real_train[all_u[ind]]) > 3 and len(real_train[all_u[ind]]) < 20:
            medium3_seq_rankeditems_list.append(sorted_ind)
            medium3_seq_test_indices.append(0)
            medium3_seq_scale_pred_list.append(scale_pred)
            medium3_seq_test_allitems.append(test_item_idx[0])

        if len(real_train[all_u[ind]]) > 7 and len(real_train[all_u[ind]]) < 20:
            medium7_seq_rankeditems_list.append(sorted_ind)
            medium7_seq_test_indices.append(0)
            medium7_seq_scale_pred_list.append(scale_pred)
            medium7_seq_test_allitems.append(test_item_idx[0])

        if len(real_train[all_u[ind]]) >= 20:
            long_seq_rankeditems_list.append(sorted_ind)
            long_seq_test_indices.append(0)
            long_seq_scale_pred_list.append(scale_pred)
            long_seq_test_allitems.append(test_item_idx[0])


        rankeditem_oneuserids = [int(test_item_idx[i]) for i in list((-1*np.array(unk_predictions)).argsort())]
        rankeditem_scores = [unk_predictions[i] for i in list((-1*np.array(unk_predictions)).argsort())]

        one_pred_result = {"u_ind": int(all_u[ind]), "u_pos_gd": int(test_item_idx[0])}
        one_pred_result["predicted"] = [int(item_id_pred) for item_id_pred in rankeditem_oneuserids[:100]]
        all_predictions_results_output.append(one_pred_result)


    batch_data = zip(rankeditems_list, test_indices, scale_pred_list, test_allitems)
    batch_result = pool.map(eval_one_interaction, batch_data)
    for re in batch_result:
        results["precision"] += re["precision"]
        results["recall"] += re["recall"]
        results["ndcg"] += re["ndcg"]
        results["hit_ratio"] += re["hit_ratio"]
        results["auc"] += re["auc"]
        results["mrr"] += re["mrr"]
    results["precision"] /= len(eval_data)
    results["recall"] /= len(eval_data)
    results["ndcg"] /= len(eval_data)
    results["hit_ratio"] /= len(eval_data)
    results["auc"] /= len(eval_data)
    results["mrr"] /= len(eval_data)
    print(f"testing #of users: {len(eval_data)}")



    short_seq_batch_data = zip(short_seq_rankeditems_list, short_seq_test_indices, short_seq_scale_pred_list, short_seq_test_allitems)
    short_seq_batch_result = pool.map(eval_one_interaction, short_seq_batch_data)
    for re in short_seq_batch_result:
        short_seq_results["precision"] += re["precision"]
        short_seq_results["recall"] += re["recall"]
        short_seq_results["ndcg"] += re["ndcg"]
        short_seq_results["hit_ratio"] += re["hit_ratio"]
        short_seq_results["auc"] += re["auc"]
        short_seq_results["mrr"] += re["mrr"]
    short_seq_results["precision"] /= len(short_seq_test_indices)
    short_seq_results["recall"] /= len(short_seq_test_indices)
    short_seq_results["ndcg"] /= len(short_seq_test_indices)
    short_seq_results["hit_ratio"] /= len(short_seq_test_indices)
    short_seq_results["auc"] /= len(short_seq_test_indices)
    short_seq_results["mrr"] /= len(short_seq_test_indices)

    print(f"testing #of short seq users with less than 3 training points: {len(short_seq_test_indices)}")



    short7_seq_batch_data = zip(short7_seq_rankeditems_list, short7_seq_test_indices, short7_seq_scale_pred_list, short7_seq_test_allitems)
    short7_seq_batch_result = pool.map(eval_one_interaction, short7_seq_batch_data)
    for re in short7_seq_batch_result:
        short7_seq_results["precision"] += re["precision"]
        short7_seq_results["recall"] += re["recall"]
        short7_seq_results["ndcg"] += re["ndcg"]
        short7_seq_results["hit_ratio"] += re["hit_ratio"]
        short7_seq_results["auc"] += re["auc"]
        short7_seq_results["mrr"] += re["mrr"]
    short7_seq_results["precision"] /= len(short7_seq_test_indices)
    short7_seq_results["recall"] /= len(short7_seq_test_indices)
    short7_seq_results["ndcg"] /= len(short7_seq_test_indices)
    short7_seq_results["hit_ratio"] /= len(short7_seq_test_indices)
    short7_seq_results["auc"] /= len(short7_seq_test_indices)
    short7_seq_results["mrr"] /= len(short7_seq_test_indices)
    print(f"testing #of short seq users with less than 7 training points: {len(short7_seq_test_indices)}")


    short37_seq_batch_data = zip(short37_seq_rankeditems_list, short37_seq_test_indices, short37_seq_scale_pred_list, short37_seq_test_allitems)
    short37_seq_batch_result = pool.map(eval_one_interaction, short37_seq_batch_data)
    for re in short37_seq_batch_result:
        short37_seq_results["precision"] += re["precision"]
        short37_seq_results["recall"] += re["recall"]
        short37_seq_results["ndcg"] += re["ndcg"]
        short37_seq_results["hit_ratio"] += re["hit_ratio"]
        short37_seq_results["auc"] += re["auc"]
        short37_seq_results["mrr"] += re["mrr"]
    short37_seq_results["precision"] /= len(short37_seq_test_indices)
    short37_seq_results["recall"] /= len(short37_seq_test_indices)
    short37_seq_results["ndcg"] /= len(short37_seq_test_indices)
    short37_seq_results["hit_ratio"] /= len(short37_seq_test_indices)
    short37_seq_results["auc"] /= len(short37_seq_test_indices)
    short37_seq_results["mrr"] /= len(short37_seq_test_indices)
    print(f"testing #of short seq users with 3 - 7 training points: {len(short37_seq_test_indices)}")



    medium3_seq_batch_data = zip(medium3_seq_rankeditems_list, medium3_seq_test_indices, medium3_seq_scale_pred_list, medium3_seq_test_allitems)
    medium3_seq_batch_result = pool.map(eval_one_interaction, medium3_seq_batch_data)
    for re in medium3_seq_batch_result:
        medium3_seq_results["precision"] += re["precision"]
        medium3_seq_results["recall"] += re["recall"]
        medium3_seq_results["ndcg"] += re["ndcg"]
        medium3_seq_results["hit_ratio"] += re["hit_ratio"]
        medium3_seq_results["auc"] += re["auc"]
        medium3_seq_results["mrr"] += re["mrr"]
    medium3_seq_results["precision"] /= len(medium3_seq_test_indices)
    medium3_seq_results["recall"] /= len(medium3_seq_test_indices)
    medium3_seq_results["ndcg"] /= len(medium3_seq_test_indices)
    medium3_seq_results["hit_ratio"] /= len(medium3_seq_test_indices)
    medium3_seq_results["auc"] /= len(medium3_seq_test_indices)
    medium3_seq_results["mrr"] /= len(medium3_seq_test_indices)
    print(f"testing #of short seq users with medium3 training points: {len(medium3_seq_test_indices)}")



    medium7_seq_batch_data = zip(medium7_seq_rankeditems_list, medium7_seq_test_indices, medium7_seq_scale_pred_list, medium7_seq_test_allitems)
    medium7_seq_batch_result = pool.map(eval_one_interaction, medium7_seq_batch_data)
    for re in medium7_seq_batch_result:
        medium7_seq_results["precision"] += re["precision"]
        medium7_seq_results["recall"] += re["recall"]
        medium7_seq_results["ndcg"] += re["ndcg"]
        medium7_seq_results["hit_ratio"] += re["hit_ratio"]
        medium7_seq_results["auc"] += re["auc"]
        medium7_seq_results["mrr"] += re["mrr"]
    medium7_seq_results["precision"] /= len(medium7_seq_test_indices)
    medium7_seq_results["recall"] /= len(medium7_seq_test_indices)
    medium7_seq_results["ndcg"] /= len(medium7_seq_test_indices)
    medium7_seq_results["hit_ratio"] /= len(medium7_seq_test_indices)
    medium7_seq_results["auc"] /= len(medium7_seq_test_indices)
    medium7_seq_results["mrr"] /= len(medium7_seq_test_indices)
    print(f"testing #of short seq users with medium7 training points: {len(medium7_seq_test_indices)}")


    long_seq_batch_data = zip(long_seq_rankeditems_list, long_seq_test_indices, long_seq_scale_pred_list, long_seq_test_allitems)
    long_seq_batch_result = pool.map(eval_one_interaction, long_seq_batch_data)
    for re in long_seq_batch_result:
        long_seq_results["precision"] += re["precision"]
        long_seq_results["recall"] += re["recall"]
        long_seq_results["ndcg"] += re["ndcg"]
        long_seq_results["hit_ratio"] += re["hit_ratio"]
        long_seq_results["auc"] += re["auc"]
        long_seq_results["mrr"] += re["mrr"]
    long_seq_results["precision"] /= len(long_seq_test_indices)
    long_seq_results["recall"] /= len(long_seq_test_indices)
    long_seq_results["ndcg"] /= len(long_seq_test_indices)
    long_seq_results["hit_ratio"] /= len(long_seq_test_indices)
    long_seq_results["auc"] /= len(long_seq_test_indices)
    long_seq_results["mrr"] /= len(long_seq_test_indices)

    print(f"testing #of short seq users with >= 20 training points: {len(long_seq_test_indices)}")
    return results, short_seq_results, short7_seq_results, short37_seq_results, medium3_seq_results, medium7_seq_results, long_seq_results, all_predictions_results_output


#def evaluate_valid(model, dataset, args, sess):
#    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
#    results = {
#            "precision": np.zeros(len(Ks)),
#            "recall": np.zeros(len(Ks)),
#            "ndcg": np.zeros(len(Ks)),
#            "hit_ratio": np.zeros(len(Ks)),
#            "auc": 0.,
#            "mrr": 0.,
#    }
#    valid_interactions = 0
#    pool = multiprocessing.Pool(cores)
#    rs = []
#    #if usernum>10000:
#    #    users = random.sample(range(1, usernum + 1), 10000)
#    #else:
#    #    users = range(1, usernum + 1)
#    users = list(valid.keys())
#    for u in tqdm(users):
#        if len(train[u]) < 1 or len(valid[u]) < 1: continue
#
#        seq = np.zeros([args.maxlen], dtype=np.int32)
#        idx = args.maxlen - 1
#        for i in reversed(train[u]):
#            seq[idx] = i
#            idx -= 1
#            if idx == -1: break
#
#        rated = set(train[u])
#        rated.add(0)
#        item_idx = copy.deepcopy(valid[u])
#        #for _ in range(100):
#        #    t = np.random.randint(1, itemnum + 1)
#        #    while t in rated: t = np.random.randint(1, itemnum + 1)
#        #    item_idx.append(t)
#        item_idx += list(set([i for i in range(itemnum)]) - rated - set(test.get(u, [])) - set(valid[u]))
#
#        gd_prob = [0 for _ in range(len(item_idx))]
#        gd_prob[0] = 1
#
#        predictions = -model.predict(sess, [u], [seq])
#        #predictions = predictions[0]
#
#        unk_predictions = []
#        for i in item_idx:
#            unk_predictions.append(predictions[0][i])
#        # print(predictions.argsort())
#        scaler = MinMaxScaler()
#        scale_pred = list(np.transpose(scaler.fit_transform(np.transpose(-1*np.array([unk_predictions]))))[0])
#
#        #rank = predictions.argsort().argsort()[0]
#        rankeditems = np.array(unk_predictions).argsort()
#        valid_indices = [ind for ind in range(len(valid[u]))]
#        valid_allitems = copy.deepcopy(valid[u])
#        rankeditems_list = [rankeditems for _ in range(len(valid[u]))]
#        scale_pred_list = [scale_pred for _ in range(len(valid[u]))]
#        valid_interactions += len(valid[u])
#        batch_data = zip(rankeditems_list, valid_indices, scale_pred_list, valid_allitems)
#        batch_result = pool.map(eval_one_interaction, batch_data)
#        for re in batch_result:
#            results["precision"] += re["precision"]
#            results["recall"] += re["recall"]
#            results["ndcg"] += re["ndcg"]
#            results["hit_ratio"] += re["hit_ratio"]
#            results["auc"] += re["auc"]
#            results["mrr"] += re["mrr"]
#    results["precision"] /= valid_interactions
#    results["recall"] /= valid_interactions
#    results["ndcg"] /= valid_interactions
#    results["hit_ratio"] /= valid_interactions
#    results["auc"] /= valid_interactions
#    results["mrr"] /= valid_interactions
#    
#    print(f"validation #of valid interactions: {valid_interactions}")
#    return results
