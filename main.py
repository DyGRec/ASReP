import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *
import traceback, sys
import json


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--evalnegsample', default=-1, type=int)
parser.add_argument('--reversed', default=0, type=int)
parser.add_argument('--reversed_gen_number', default=-1, type=int)
parser.add_argument('--M', default=10, type=int)
parser.add_argument('--reversed_pretrain', default=-1, type=int)
parser.add_argument('--aug_traindata', default=-1, type=int)

args = parser.parse_args()

dataset = data_load(args.dataset, args)
[user_train, user_valid, user_test, original_train, usernum, itemnum] = dataset
num_batch = int(len(user_train) / args.batch_size)
cc = []
for u in user_train:
    cc.append(len(user_train[u]))
cc = np.array(cc)
print('average sequence length: %.2f' % np.mean(cc))
print('min seq length: %.2f' % np.min(cc))
print('max seq length: %.2f' % np.max(cc))
print('quantile 25 percent: %.2f' % np.quantile(cc, 0.25))
print('quantile 50 percent: %.2f' % np.quantile(cc, 0.50))
print('quantile 75 percent: %.2f' % np.quantile(cc, 0.75))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = Model(usernum, itemnum, args)
#sess.run(tf.global_variables_initializer())

aug_data_signature = './aug_data/{}/lr_{}_maxlen_{}_hsize_{}_nblocks_{}_drate_{}_l2_{}_nheads_{}_gen_num_{}_M_{}'.format(args.dataset, args.lr, args.maxlen, args.hidden_units, args.num_blocks, args.dropout_rate, args.l2_emb, args.num_heads, args.reversed_gen_number, args.M)
print(aug_data_signature)

model_signature = 'lr_{}_maxlen_{}_hsize_{}_nblocks_{}_drate_{}_l2_{}_nheads_{}_gen_num_{}'.format(args.lr, args.maxlen, args.hidden_units, args.num_blocks, args.dropout_rate, args.l2_emb, args.num_heads, 5)

if not os.path.isdir('./aug_data/'+args.dataset):
    os.mkdir('./aug_data/'+args.dataset)

saver = tf.train.Saver()
if args.reversed_pretrain == -1:
    sess.run(tf.global_variables_initializer())
else:
    saver.restore(sess, './reversed_models/'+args.dataset+'_reversed/'+model_signature+'.ckpt')
    print('pretain model loaded')

T = 0.0
t0 = time.time()

try:
    for epoch in range(1, args.num_epochs + 1):

        #print(num_batch)
        for step in range(num_batch):#tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        #for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True})
        print('epoch: %d, loss: %8f' % (epoch, loss))

        #tvars = tf.trainable_variables()
        #tvars_vals = sess.run(tvars, {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
        #                              model.is_training: False})
        #i_epoch_file = open(var_directory_path + str(epoch) + '.txt', 'w')
        #for var, val in zip(tvars, tvars_vals):
        #    i_epoch_file.write("{}\n{}\n".format(var.name, val))
        #i_epoch_file.close()

        if (epoch % 20 == 0 and epoch >= 200) or epoch == args.num_epochs:
            t1 = time.time() - t0
            T += t1
            print("start testing")
            RANK_RESULTS_DIR = f"./rank_results/{args.dataset}_pretain_{args.reversed_pretrain}"
            if not os.path.isdir(RANK_RESULTS_DIR):
                os.mkdir(RANK_RESULTS_DIR)
            rank_test_file = RANK_RESULTS_DIR + '/' + model_signature + '_predictions.json'
            t_test, t_test_short_seq, t_test_short7_seq, t_test_short37_seq, t_test_medium3_seq, t_test_medium7_seq, t_test_long_seq, test_rankitems = evaluate(model, dataset, args, sess, "test")
            #if args.reversed == 0:
            #    with open(rank_test_file, 'w') as f:
            #        for eachpred in test_rankitems:
            #            f.write(json.dumps(eachpred) + '\n')
            if not (args.reversed == 1):
                t_valid, t_valid_short_seq, t_valid_short7_seq, t_valid_short37_seq, t_valid_medium3_seq, t_valid_medium7_seq, t_valid_long_seq, valid_rankitems = evaluate(model, dataset, args, sess, "valid")
            #print('epoch:%d, time: %f(s), valid NDCG@10: %.4f HR@10: %.4f NDCG@1: %.4f HR@1: %.4f NDCG@5: %.4f HR@5: %.4f test NDCG@10: %.4f HR@10: %.4f NDCG@1: %.4f HR@1: %.4f NDCG@5: %.4f HR@5: %.4f' % (
            #epoch, T, t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_valid[4], t_valid[5], t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5]))
                print('epoch: '+ str(epoch)+' validationall: '+str(t_valid) + '\nepoch: '+str(epoch)+' testall: ' + str(t_test))
                print('epoch: '+ str(epoch)+' validationshort: '+str(t_valid_short_seq) + '\nepoch: '+str(epoch)+' testshort: ' + str(t_test_short_seq))
                print('epoch: '+ str(epoch)+' validationshort7: '+str(t_valid_short7_seq) + '\nepoch: '+str(epoch)+' testshort7: ' + str(t_test_short7_seq))
                print('epoch: '+ str(epoch)+' validationshort37: '+str(t_valid_short37_seq) + '\nepoch: '+str(epoch)+' testshort37: ' + str(t_test_short37_seq))
                print('epoch: '+ str(epoch)+' validationmedium3: '+str(t_valid_medium3_seq) + '\nepoch: '+str(epoch)+' testmedium3: ' + str(t_test_medium3_seq))
                print('epoch: '+ str(epoch)+' validationmedium7: '+str(t_valid_medium7_seq) + '\nepoch: '+str(epoch)+' testmedium7: ' + str(t_test_medium7_seq))
                print('epoch: '+ str(epoch)+' validationlong: '+str(t_valid_long_seq) + '\nepoch: '+str(epoch)+' testlong: ' + str(t_test_long_seq))
            else:
                print('epoch: '+str(epoch)+' test: ' + str(t_test))

            t0 = time.time()
except Exception as e:
    print(e)
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
    sampler.close()
    exit(1)

if args.reversed == 1:
    augmented_data = data_augment(model, dataset, args, sess, args.reversed_gen_number)
    with open(aug_data_signature+'.txt', 'w') as f:
        for u, aug_ilist in augmented_data.items():
            for ind, aug_i in enumerate(aug_ilist):
                f.write(str(u-1) + '\t' + str(aug_i - 1) + '\t' + str(-(ind+1)) + '\n')
    if args.reversed_gen_number > 0:
        if not os.path.exists('./reversed_models/'+args.dataset+'_reversed/'):
            os.makedirs('./reversed_models/'+args.dataset+'_reversed/')
        saver.save(sess, './reversed_models/'+args.dataset+'_reversed/'+model_signature+'.ckpt')
sampler.close()
