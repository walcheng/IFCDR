import torch.nn as nn
import torch


class BPR(nn.Module):
	def __init__(self, user_num, item_num, factor_num):
		super(BPR, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""
		self.embed_user = nn.Embedding(user_num, factor_num)
		self.embed_item = nn.Embedding(item_num, factor_num)

		nn.init.normal_(self.embed_user.weight, std=0.01)
		nn.init.normal_(self.embed_item.weight, std=0.01)
		self.f = nn.Sigmoid()

	def get_user_rating(self, user, item_i, item_j):
		user = self.embed_user(user)
		item_i = self.embed_item(item_i)
		item_j = self.embed_item(item_j)
		prediction_i = (user * item_i).sum(dim=-1)
		prediction_j = (user * item_j).sum(dim=-1)
		return prediction_i, prediction_j

	def forward(self, user, item_i, item_j):
		users_emb = self.embed_user(user)
		pos_emb = self.embed_item(item_i)
		neg_emb = self.embed_item(item_j)

		pos_scores = torch.sum(users_emb * pos_emb, dim=1)
		neg_scores = torch.sum(users_emb * neg_emb, dim=1)
		loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
		reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
							  pos_emb.norm(2).pow(2) +
							  neg_emb.norm(2).pow(2)) / float(len(user))
		return loss, reg_loss