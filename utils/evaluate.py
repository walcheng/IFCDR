import numpy as np
import torch
import tqdm


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def mrr(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return 1/(index+1)
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def test_single(device, model, test_loader, top_k):
	HR, NDCG, MRR = [], [], []
	print('testing...')
	model.eval()
	for user, item_i, item_j in tqdm.tqdm(test_loader, smoothing=0, mininterval=1.0):
		user = user.to(device)
		item_i = item_i.to(device)
		item_j = item_j.to(device)  # not useful when testing

		prediction_i, prediction_j = model.get_user_rating(user, item_i, item_j)
		_, indices = torch.topk(prediction_i, top_k)
		recommends = torch.take(
				item_i, indices).cpu().numpy().tolist()

		gt_item = item_i[0].item()  # negative sample 中第一个为 positive sample
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))
		MRR.append(mrr(gt_item, recommends))
	return np.mean(HR), np.mean(NDCG), np.mean(MRR)