from tqdm import tqdm
import torch
import numpy as np
import time
import utils.evaluate as evaluate


def loss_combinate(sim_loss, dis_loss, rec_loss, dom_loss, loss_type):
    if loss_type == 'full':
        return sim_loss * 0.1 + dis_loss * 0.1 + rec_loss * 0.6 + dom_loss * 0.4
    elif loss_type == 'no_dis':
        return sim_loss * 0.2 + rec_loss * 0.8
    elif loss_type == 'single':
        return sim_loss

def train_cross_model(model, param, loss_type='full'):
    """
    param = run_config = "train_loader": train_cross_batch,
                      "valid_loader": valid_cross_batch,
                      "opt": opt,
                      "device": args.device,
                      "early_stop": early_stop_pre,
                      "epoch": args.epoch}
    """
    model.train()
    for epoch in range(param['epoch']):
        train_loss = []
        pbar = tqdm(param["train_loader"])
        pbar.set_description(f'[Train epoch {epoch}]')
        for data_batch, y in pbar:
            data_batch = data_batch.to(param['device'])
            model.zero_grad(set_to_none=True)
            sim_loss, dis_loss, rec_loss, dom_loss = model.cross_forward(data_batch)
            if loss_type != None:
                loss = loss_combinate(sim_loss, dis_loss, rec_loss, dom_loss, loss_type)
            else:
                loss = rec_loss
            loss.backward()
            param['opt'].step()
            pbar.set_postfix(loss=loss.item(), sim_loss=sim_loss.item(),
                             dis_loss=dis_loss.item(), rec_loss=rec_loss.item(),
                             dom_loss=dom_loss.item())
            train_loss.append(loss.item())

        print(" *** Evaluation ...")
        loss_v = valid_cross_model(model, param, loss_type=loss_type)
        param["early_stop"](loss_v, model)
        if param["early_stop"].early_stop:
            print(" ****** Early stopping ...")
            return 1
    return 0


def valid_cross_model(model, param, loss_type='full'):
    model.eval()
    valid_loss = []
    with torch.no_grad():
        pbar = tqdm(param["valid_loader"])
        for data_batch, y in pbar:
            data_batch = data_batch.to(param['device'])
            sim_loss, dis_loss, rec_loss, dom_loss = model.cross_forward(data_batch)
            if loss_type != None:
                loss = loss_combinate(sim_loss, dis_loss, rec_loss, dom_loss, loss_type)
            else:
                loss = rec_loss
            valid_loss.append(loss.item())
    return np.mean(valid_loss)


def tune_cross_model(model, param):
    model.train()
    count, best_hr = 0, 0
    start_time = time.time()
    for epoch in range(param['epoch']):
        param['train_loader'].dataset.ng_sample()
        print('domain {} training...'.format(param['target_domain']))
        for user, item_i, item_j in tqdm(param['train_loader'], smoothing=0, mininterval=1.0):
            user = user.to(param['device'])
            item_i = item_i.to(param['device'])
            item_j = item_j.to(param['device'])

            model.zero_grad()
            loss, reg_loss = model(user, item_i, item_j)
            loss = loss + reg_loss * param['decay']
            loss.backward()
            param['opt'].step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count += 1

        model.eval()
        HR, NDCG, MRR = evaluate.test_single(param['device'], model, param['valid_loader'], param['top_k'])
        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:04d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("domain {}, HR: {:.4f}\tNDCG: {:.4f}\tMRR: {:.4f}".format(param['target_domain'], np.mean(HR),
              np.mean(NDCG), np.mean(MRR)))