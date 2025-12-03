import os
import time
import argparse
import numpy as np
import random
import torch
import json
import tqdm
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils.bpr as bpr
import utils.evaluate as evaluate
import utils.dataloader as data_loader
from utils.base_tools import log_writer


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='./model/')
parser.add_argument("--task", type=str, default='1', help='training model for different cross-domain task')
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=0, help="weight decay for l2 regularization")  # 1e-4
parser.add_argument("--batch_size", type=int, default=2048, help="batch size for training")
parser.add_argument("--epochs", type=int, default=30, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--latent_dim", type=int, default=64, help="predictive factors numbers in the model")
parser.add_argument("--num_ng", type=int, default=3, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sampled negative items for testing")
parser.add_argument("--out", default=True, help="save model or not")
parser.add_argument("--train", default=True, help="train model")
parser.add_argument("--seed", default=2020)
parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
parser.add_argument('--record_log', default=True)
parser.add_argument('--log_file', default='results/single_log.txt', help='log file')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

with open('config.json', 'r') as f:
	config = json.load(f)
cross_dom = ['idom', 'jdom', 'kdom']

for task in ['1']:
	data_root = config['root'] + 'ready/' + str(int(config['ratio'][0] * 10)) + '_' + \
				str(int(config['ratio'][1] * 10)) + '/task_' + args.task

	for dom in ['kdom']:
		#  PREPARE DATASET #####################################
		neg_sample_dom = config['cross_domain_targets'][task][dom]
		user_num = config['cross_domain_targets'][task]['uid_'+dom]
		item_num = config['cross_domain_targets'][task]['iid_'+dom]
		train_path = data_root + '/' + neg_sample_dom + '_train_single.csv'  # normal train data, all are pos sample
		test_path = data_root + '/' + neg_sample_dom + '_test.negative.txt'  # each line is one pos pairs, 100 neg pairs
		train_data, test_data, train_mat = data_loader.load_all(train_path, test_path, user_num, item_num)

		# construct the train and test datasets
		train_dataset = data_loader.BPRData(
				train_data, item_num, train_mat, args.num_ng, True)
		test_dataset = data_loader.BPRData(
				test_data, item_num, train_mat, 0, False)
		train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
		test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng+1, shuffle=False)

		# CREATE MODEL #####################################
		model = bpr.BPR(user_num, item_num, args.latent_dim)
		model.to(device)
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

		# writer = SummaryWriter() # for visualization

		# TRAINING #####################################
		count, best_hr = 0, 0
		if args.train:
			for epoch in range(args.epochs):
				model.train()
				start_time = time.time()
				train_loader.dataset.ng_sample()
				print('domain {} training...'.format(neg_sample_dom))
				for user, item_i, item_j in tqdm.tqdm(train_loader, smoothing=0, mininterval=1.0):
					user = user.to(device)
					item_i = item_i.to(device)
					item_j = item_j.to(device)

					model.zero_grad()
					loss, reg_loss = model(user, item_i, item_j)
					loss = loss + reg_loss * args.weight_decay
					loss.backward()
					optimizer.step()
					# writer.add_scalar('data/loss', loss.item(), count)
					count += 1

				model.eval()
				HR, NDCG, MRR = evaluate.test_single(device, model, test_loader, args.top_k)

				elapsed_time = time.time() - start_time
				print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
						time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
				print("domain {}, HR: {:.3f}\tNDCG: {:.3f}\tMRR: {:.3f}".format(neg_sample_dom, np.mean(HR),
																				np.mean(NDCG), np.mean(MRR)))

				if HR > best_hr:
					best_hr, best_ndcg, best_mrr, best_epoch = HR, NDCG, MRR, epoch
					if args.out:
						model_path = args.model_path + 'task_' + args.task + '/'
						if not os.path.exists(model_path):
							os.mkdir(model_path)
						torch.save(model.state_dict(), '{}{}_BPR.pt'.format(model_path, neg_sample_dom))
			print("End. Best epoch {:03d}: HR = {:.6f}, \
						NDCG = {:.6f}, MRR = {:.6f}".format(best_epoch, best_hr, best_ndcg, best_mrr))
			if args.record_log:
				log_writer(args.log_file, "End. domain {}. latent dim {}.  Best epoch {:03d}: HR = {:.4f}, \
					NDCG = {:.4f}, MRR= {:.4f} ".format(neg_sample_dom, args.latent_dim, best_epoch, best_hr, best_ndcg, best_mrr))
		else:
			model_path = args.model_path + 'task_' + args.task + '/'
			model.load_state_dict(torch.load('{}{}_BPR.pt'.format(model_path, neg_sample_dom)))
			model.eval()
			HR, NDCG, MRR = evaluate.test_single(device, model, test_loader, args.top_k)
			print("End. domain {}: HR = {:.6f}, \
									NDCG = {:.6f}, MRR = {:.6f}".format(neg_sample_dom, HR, NDCG, MRR))

