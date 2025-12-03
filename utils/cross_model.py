import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils.base_tools as base_tools


class CrossDomainBase(nn.Module):
    # 1. 能获取训练好domain的user embedding
    def __init__(self, config):
        super().__init__()
        self.target_dirs = config['target_dirs']
        self.latent_dim = config['latent_dim']
        self.num_domains = len(config['single_dirs'])
        self.device = config['device']
        self.domain_dict = config['domain_dict']
        self.target_domain = config['target_domain']
        self.batch_size = config['batch_size']
        self.final_embed = config['final_embed']
        self.dropout = config['dropout']
        #  loss init
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss()
        # load the user embeddings from all domains
        user_embedding_list = []
        for i in range(self.num_domains):
            print(f"Loading pre-trained user embedding for domain {config['single_dirs'][i]}")
            model_dict = torch.load(config['single_dirs'][i], map_location=config['device'])
            # user embedding table sizes
            user_sizes = model_dict['embed_user.weight'].size()
            user_embedding_list.append(torch.nn.Embedding(user_sizes[0], user_sizes[1],
                                                          _weight=model_dict['embed_user.weight']))
        self.user_embedding_list = nn.ModuleList(user_embedding_list)

        # get target item embedding
        model_dict = torch.load(config['target_dirs'], map_location=config['device'])
        item_sizes = model_dict['embed_item.weight'].size()
        self.target_item_embedding = torch.nn.Embedding(item_sizes[0], item_sizes[1],
                                                        _weight=model_dict['embed_item.weight'])
        self.fix_user = config['fix_user']
        self.fix_item = config['fix_item']
        self.dom_dis_loss = config['dom_dis_loss']
        self.mix_up = config['mix_up']
        self.f = nn.Sigmoid()

    def forward(self, users, item_i, item_j):
        users_emb = self.get_user_embed(users)
        if self.fix_item:
            with torch.no_grad():
                pos_emb = self.target_item_embedding(item_i.long())
                neg_emb = self.target_item_embedding(item_j.long())
        else:
            pos_emb = self.target_item_embedding(item_i.long())
            neg_emb = self.target_item_embedding(item_j.long())

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        # reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
        #                       pos_emb.norm(2).pow(2) +
        #                       neg_emb.norm(2).pow(2)) / float(len(users))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def get_user_rating(self, user, item_i, item_j):
        users_emb = self.get_user_embed(user)
        if self.fix_item:
            with torch.no_grad():
                pos_emb = self.target_item_embedding(item_i.long())
                neg_emb = self.target_item_embedding(item_j.long())
        else:
            pos_emb = self.target_item_embedding(item_i.long())
            neg_emb = self.target_item_embedding(item_j.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        return pos_scores, neg_scores


class CrossDomainBase1(CrossDomainBase):
    """
    1: A shared FC layer
    Create domain specific encoder
    Create domain common encoder, with mixup mechanism to get more generalized
    create domain discriminator
    create the loss via a soft subspace orthogonality constraint between the private and shared representation (DSN)
    """
    def __init__(self, config):
        super().__init__(config)
        # load the user embeddings from all domains
        user_embedding_list_fix = []
        for i in range(self.num_domains):
            print(f"Loading pre-trained user embedding for domain {config['single_dirs'][i]}")
            model_dict = torch.load(config['single_dirs'][i], map_location=config['device'])
            # user embedding table sizes
            user_sizes = model_dict['embed_user.weight'].size()
            user_embedding_list_fix.append(torch.nn.Embedding(user_sizes[0], user_sizes[1],
                                                              _weight=model_dict['embed_user.weight']))
        self.user_embedding_list_fix = nn.ModuleList(user_embedding_list_fix)

        # 1: A shared FC layer(mapping embeds into same space)
        self.encoder_mapping = nn.Linear(self.latent_dim, self.latent_dim)

        # 2： create domain specific encoder
        self.spe_encoder_i = nn.Sequential(
            nn.Linear(self.latent_dim, int(self.latent_dim / 2)),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.latent_dim / 2), int(self.latent_dim / 4)))
        self.spe_encoder_j = nn.Sequential(
            nn.Linear(self.latent_dim, int(self.latent_dim / 2)),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.latent_dim / 2), int(self.latent_dim / 4)))
        self.spe_encoder_k = nn.Sequential(
            nn.Linear(self.latent_dim, int(self.latent_dim / 2)),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.latent_dim / 2), int(self.latent_dim / 4)))

        # 3: create domain common encoder
        self.com_encoder = nn.Sequential(
            nn.Linear(self.latent_dim, int(self.latent_dim / 2)),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.latent_dim / 2), int(self.latent_dim / 4)))

        # create decoder
        self.decoder = nn.Sequential(
            nn.Linear(int(self.latent_dim / 4), int(self.latent_dim / 2)),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.latent_dim / 2), self.latent_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.latent_dim, int(self.latent_dim)))
        # self.decoder = nn.Sequential(
        #     nn.Linear(int(self.latent_dim / 8), int(self.latent_dim / 4)),
        #     nn.PReLU(),
        #     nn.Linear(int(self.latent_dim / 4), int(self.latent_dim / 2)),
        #     nn.PReLU(),
        #     nn.Linear(int(self.latent_dim / 2), self.latent_dim),
        #     nn.PReLU(),
        #     nn.Linear(self.latent_dim, int(self.latent_dim)))

        # domain discriminator
        self.domain_fc = nn.Linear(int(self.latent_dim / 4), self.num_domains)

        if self.dom_dis_loss != 'cross-entropy':
            self.angle_margin_loss = base_tools.AngularPenaltySMLoss(int(self.latent_dim / 4), self.num_domains + 1, loss_type=self.dom_dis_loss)

        # the predictor
        self.general_embedding_adapt = nn.Linear(int(self.latent_dim / 4), self.latent_dim)
        self.spe_embedding_adapt = nn.Linear(int(self.latent_dim / 4), self.latent_dim)


    def cross_forward(self, users):
        # get embeddings
        embeds = self.get_org_embed(users)
        embeds = self.encoder_mapping(embeds)
        emb_i, emb_j, emb_k, emb_list = self.split_embed(embeds)

        # forward specific encoder
        sep_emb_i = self.spe_encoder_i(emb_i)
        sep_emb_j = self.spe_encoder_j(emb_j)
        sep_emb_k = self.spe_encoder_k(emb_k)

        # aug the embeddings with mix-up
        target_dom_idx = self.domain_dict[self.target_domain]
        mixed_emb_list = self.mixup_embed(emb_list, target_dom_idx, self.mix_up)

        # forward common encoder
        com_hid_embed_list = []
        for i in range(len(mixed_emb_list)):
            com_hid_embed_list.append(self.com_encoder(mixed_emb_list[i]))
        # similarity loss
        sim_loss_list = base_tools.sim_loss(com_hid_embed_list)

        com_hid_embed = torch.mean(torch.stack(com_hid_embed_list, dim=0), dim=0)

        # reconstruction loss
        latent_i = torch.add(sep_emb_i, com_hid_embed)
        latent_i = self.decoder(latent_i)
        latent_j = torch.add(sep_emb_j, com_hid_embed)
        latent_j = self.decoder(latent_j)
        latent_k = torch.add(sep_emb_k, com_hid_embed)
        latent_k = self.decoder(latent_k)

        rec_loss_i = self.mse(latent_i, emb_i)
        rec_loss_j = self.mse(latent_j, emb_j)
        rec_loss_k = self.mse(latent_k, emb_k)

        # Disparity loss
        dis_loss_list = base_tools.dis_loss(sep_emb_i, sep_emb_j, sep_emb_k)

        # forward domain discriminator
        if self.dom_dis_loss == 'cross-entropy':
            sep_scores_i = self.domain_fc(sep_emb_i)
            sep_scores_j = self.domain_fc(sep_emb_j)
            sep_scores_k = self.domain_fc(sep_emb_k)
            com_scores = self.domain_fc(com_hid_embed_list[0])

            # ground-truth for domain discriminator
            sep_target_i = torch.zeros(users.size(0)).long()
            sep_target_i = F.one_hot(sep_target_i, num_classes=3).float().cuda()  # [1,0,0]
            sep_target_j = torch.ones(users.size(0)).long()
            sep_target_j = F.one_hot(sep_target_j, num_classes=3).float().cuda()  # [0,1,0]
            sep_target_k = (torch.ones(users.size(0))*2).long()
            sep_target_k = F.one_hot(sep_target_k, num_classes=3).float().cuda()  # [0,0,1]
            com_target = Variable(torch.ones(users.size(0), 3)/3.0).cuda()  # [0.33,0.33,0.33]

            # domain discriminator loss
            dom_sep_loss_i = self.ce(F.log_softmax(sep_scores_i, dim=1), sep_target_i)
            dom_sep_loss_j = self.ce(F.log_softmax(sep_scores_j, dim=1), sep_target_j)
            dom_sep_loss_k = self.ce(F.log_softmax(sep_scores_k, dim=1), sep_target_k)
            dom_com_loss = self.kld(F.log_softmax(com_scores, dim=1), com_target)
            dom_loss = (dom_sep_loss_i + dom_sep_loss_j + dom_sep_loss_k + dom_com_loss)/4
        else:

            sep_target_i = torch.tensor([0]*sep_emb_i.size(dim=0))
            sep_target_j = torch.tensor([1]*sep_emb_j.size(dim=0))
            sep_target_k = torch.tensor([2]*sep_emb_k.size(dim=0))
            com_target = torch.tensor([3]*sep_emb_i.size(dim=0))
            dom_sep_loss_i = self.angle_margin_loss(sep_emb_i,sep_target_i)
            dom_sep_loss_j = self.angle_margin_loss(sep_emb_j,sep_target_j)
            dom_sep_loss_k = self.angle_margin_loss(sep_emb_k,sep_target_k)
            dom_com_loss_i = self.angle_margin_loss(com_hid_embed_list[0],com_target)
            dom_com_loss_j = self.angle_margin_loss(com_hid_embed_list[1],com_target)
            dom_com_loss_k = self.angle_margin_loss(com_hid_embed_list[2],com_target)
            dom_loss = (dom_sep_loss_i + dom_sep_loss_j + dom_sep_loss_k +
                         dom_com_loss_i + dom_com_loss_j + dom_com_loss_k)/6

        # overall loss
        sim_loss = (sim_loss_list[0] + sim_loss_list[1] + sim_loss_list[2])/3.0
        dis_loss = (dis_loss_list[0] + dis_loss_list[1] + dis_loss_list[2])/3.0
        rec_loss = rec_loss_i + rec_loss_j + rec_loss_k

        return sim_loss, dis_loss, rec_loss, dom_loss

    def get_org_embed(self, users):
        with torch.no_grad():
            embeds = []
            for i in range(self.num_domains):
                # embeds.append(self.user_embedding_list_fix[i](users.to('cpu')))
                embeds.append(self.user_embedding_list_fix[i](users))
        embeds = torch.stack(embeds, 1)
        return embeds.to(self.device)

    def get_user_embed(self, users):
        # get common information
        target_embed = []
        embeds = self.get_org_embed(users)
        ###

        # aug the embeddings with mix-up
        target_dom_idx = self.domain_dict[self.target_domain]
        ###
        with torch.no_grad():
            embeds = self.encoder_mapping(embeds)
            emb_i, emb_j, emb_k, emb_list = self.split_embed(embeds)
            # aug the embeddings with mix-up
            target_dom_idx = self.domain_dict[self.target_domain]
            mixed_emb_list = self.mixup_embed(emb_list, target_dom_idx, self.mix_up)
            # feed-forward into the secific encoder
            sep_emb = []
            sep_emb.append(self.spe_encoder_i(emb_i))
            sep_emb.append(self.spe_encoder_j(emb_j))
            sep_emb.append(self.spe_encoder_k(emb_k))

            # average the information when go through the encoder
            com_hid_embed_list = []
            for i in range(len(mixed_emb_list)):
                com_hid_embed_list.append(self.com_encoder(mixed_emb_list[i]))
            domain_common_info = torch.mean(torch.stack(com_hid_embed_list, dim=0), dim=0)
            
        #feed-forward common and specific adapter
        domain_common_info = self.general_embedding_adapt(domain_common_info)
        domain_sepcfic_info = self.spe_embedding_adapt(sep_emb[target_dom_idx])

        # get target domain embeds
        target_dom_idx = self.domain_dict[self.target_domain]
        if self.fix_user:
            with torch.no_grad():
                target_embed.append(self.user_embedding_list[target_dom_idx](users))
        else:
            target_embed.append(self.user_embedding_list[target_dom_idx](users))
        target_embed = torch.squeeze(torch.stack(target_embed, 1))

        if self.final_embed == 'all':
            return torch.squeeze(domain_common_info + target_embed + domain_sepcfic_info)
        if self.final_embed == 'common':
            return torch.squeeze(domain_common_info + target_embed)
        if self.final_embed == 'specific':
            return torch.squeeze(target_embed + domain_sepcfic_info)
        if self.final_embed == 'stay':
            return torch.squeeze(target_embed)

    @staticmethod
    def split_embed(embeds):
        # split embeds into three parts, for each specific encoder and easy to mixup
        emb_i = embeds[:, 0, :]
        emb_j = embeds[:, 1, :]
        emb_k = embeds[:, 2, :]
        emb_list = [emb_i, emb_j, emb_k]
        return emb_i, emb_j, emb_k, emb_list

    @staticmethod
    def mixup_embed(emb_list, target_dom_idx, mix_up, alpha=100.0):
        """
       Generate three types of mixed embeds
       1: target + oth_domain1
       2: target + oth_domain2
       3: target + oth_domain1 + oth_domain2
        """
        combinations = [[0, 1], [0, 2], [1, 2]]
        mixed_emb_list = []
        lam_list = []
        if mix_up:
            if alpha > 0:
                for i in range(len(emb_list)):
                    lam = np.random.beta(alpha, alpha)
                    mix_emb = lam * emb_list[combinations[i][0]] + (1 - lam) * emb_list[combinations[i][1]]
                    mixed_emb_list.append(mix_emb)   

                # 1: target + oth_domain1
                # 2: target + oth_domain2

                # target_emb = emb_list.pop(target_dom_idx)
            #     for i in range(len(emb_list)):
            #         lam = np.random.beta(alpha, alpha)
            #         mix_emb = lam * emb_list[i] + (1 - lam) * target_emb
            #         mixed_emb_list.append(mix_emb)
            #         lam_list.append(lam)
            #     lam_list.append(np.random.beta(alpha, alpha))
            #     lam_list = np.exp(lam_list) / np.sum(np.exp(lam_list), axis=0)
            # else:
            #     lam_list = [0.33, 0.33, 0.33]

            # # 3: target + oth_domain1 + oth_domain2
            # mix_emb = lam_list[0] * emb_list[0] + lam_list[1] * emb_list[1] + lam_list[2] * target_emb
            # mixed_emb_list.append(mix_emb)
            return mixed_emb_list
        else:
            return emb_list

