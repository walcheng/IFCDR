"""
py functions: 1: read data for cross-model training;
              2: training cross-model
              3: testing cross-model
"""
import sys
import argparse
import os
import json
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import utils.base_tools as base_tools
import utils.dataloader as dataloader
import torch.optim as optim
import run_functions as run_func
import utils.cross_model as cross_model


def main(print_p=True):
    cross_model_dir = f"Cross-ld{args.latent_dim}"
    # final model tune on the specific task
    final_model_path = f"{args.model_dir}/{cross_model_dir}/best_model.pt"
    # pretrained model train common attributes
    pre_model_path = f"{args.model_dir}/{cross_model_dir}/{args.target_domain}_pre_model.pt"
    print(final_model_path)
    if not os.path.exists(f"{args.model_dir}/{cross_model_dir}"):
        os.makedirs(f"{args.model_dir}/{cross_model_dir}")

    # load all cross-domain dataset
    task = config['cross_domain_targets'][args.task][args.target_domain]
    data_root = config['root'] + 'ready/' + str(int(config['ratio'][0] * 10)) + '_' + \
                str(int(config['ratio'][1] * 10)) + '/task_' + args.task + '/' + task
    train_cross_loader, valid_cross_loader, train_adapt_loader, test_loader = \
        dataloader.cross_domain_loader(data_root, config, args, train=True)

    train_cross_batch = DataLoader(train_cross_loader, batch_size=args.cross_batch_size, shuffle=True)
    valid_cross_batch = DataLoader(valid_cross_loader, batch_size=200, shuffle=True)
    train_adapt_batch = DataLoader(train_adapt_loader, batch_size=args.tune_batch_size, shuffle=True)
    test_batch = DataLoader(test_loader, batch_size=args.test_num_ng + 1, shuffle=False)

    # build model
    domain_dict = {'idom': 0, 'jdom': 1, 'kdom': 2}
    model_config = {'target_dirs': args.target_dirs,
                    'single_dirs': args.single_dirs,
                    'latent_dim': args.latent_dim,
                    'device': args.device,
                    'dropout': args.dropout,
                    'final_embed': args.final_embed,
                    'fix_user': args.fix_user,
                    'fix_item': args.fix_item,
                    'dom_dis_loss': args.dom_dis_loss,
                    'mix_up': args.mix_up,
                    'domain_dict': domain_dict,
                    'target_domain': args.target_domain,
                    'batch_size': args.cross_batch_size}
    model = cross_model.CrossDomainBase1(model_config)

    if print_p:  # show model parameters
        for name, param in model.named_parameters():
            print(name, param.requires_grad, param.data.size())
    model = model.to(args.device)
    if args.pre_train:
        early_stop_pre = base_tools.EarlyStopping(patience=2, verbose=True, path=pre_model_path)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f'**** Starting pretraining on the cross model, task : {args.task}, domain: {task}')
        run_config = {"train_loader": train_cross_batch,
                      "valid_loader": valid_cross_batch,
                      "opt": opt,
                      "device": args.device,
                      "early_stop": early_stop_pre,
                      "epoch": args.epoch}
        if not os.path.isfile(pre_model_path):
            print(" === pre-training of Autoencoder ... ")
            run_func.train_cross_model(model, run_config, loss_type=args.encoder_loss)
            print(" Pretraining Done ")
            sys.exit()
        else:
            print(" === Loading pre-trained cross model ... ")
            early_stop = base_tools.EarlyStopping(patience=2, verbose=True, path=final_model_path)
            model_dict = model.state_dict()
            pre_dict = torch.load(pre_model_path)
            # copy all parameters instead of the embedding
            pre_dict = {k: v for k, v in pre_dict.items() if 'embedding' not in k}
            model_dict.update(pre_dict)
            model.load_state_dict(model_dict)
        # tune on cross-domain recommendation task.
        run_config['opt'].param_groups[0]['lr'] = args.tune_lr
        run_config["train_loader"] = train_adapt_batch
        run_config["valid_loader"] = test_batch
        run_config['early_stop'] = early_stop
        run_config['target_domain'] = args.target_domain
        run_config['decay'] = args.weight_decay
        run_config['top_k'] = args.top_k
        print(" ===tune multi-target cross-domain recommendation ... ")
        run_func.tune_cross_model(model, run_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Disentangled Cross Model Training')
    parser.add_argument('--cross_batch_size', type=int, default=256, help='batch size for cross training')
    parser.add_argument('--tune_batch_size', type=int, default=2048, help='batch size for tuning')
    parser.add_argument('--task', type=str, default='1', help='cross-domain from config.json')
    parser.add_argument('--target_domain', type=str, default='jdom',
                        choices=['idom', 'jdom', 'kdom'], help='target domain to be promote')
    parser.add_argument('--epoch', type=int, default=150, help='number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')  # 1e-3
    parser.add_argument('--tune_lr', type=float, default=5e-4, help='learning rate for tuning pretrained cross model')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for l2 regularizer') # 1e-4
    parser.add_argument('--dropout', type=float, default=0, help='drop out for cross model')
    parser.add_argument('--latent_dim', type=int, default=64, help='dimensions')
    parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
    parser.add_argument('--n_worker', type=int, default=4, help='number of workers for the train data loader')
    parser.add_argument('--train_num_ng', type=int, default=3, help='sampled negative items for training')
    parser.add_argument("--test_num_ng", type=int, default=99, help="sampled negative items for testing")
    parser.add_argument('--fix_item', type=bool, default=False,
                        help='whether to fix item embeddings in the target domain')
    parser.add_argument('--fix_user', type=bool, default=False,
                        help='whether to fix user embeddings in the target domain')
    parser.add_argument('--dom_dis_loss', type=str, default='arcface', choices=['cross-entropy', 'arcface', 'cosface','sphereface'],
                        help='whether use the mix up strategy in cross model training')
    parser.add_argument('--mix_up', type=bool, default=True,
                        help='whether use the mix up strategy in cross model training')
    parser.add_argument('--pre_train', type=bool, default=True, help='if pretrain cross model parameters')
    parser.add_argument('--pre_train_valid', type=bool, default=True, help='if valid the pretrained cross model')
    parser.add_argument("--gpu", type=str, default="1", help="gpu card ID")
    parser.add_argument('--final_embed', type=str, default='all', choices=['all', 'common','specific'],
                        help='what to transfer: Domain-independent representation(d_in),domain-specific (d_sp) or both')
    parser.add_argument('--encoder_loss', type=str, default='full', help='loss to train the encoder, full or rec ')
    parser.add_argument("--seed", default=2020)

    args = parser.parse_args()
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    with open('config.json', 'r') as f:
        config = json.load(f)
    cross_dom = ['idom', 'jdom', 'kdom']
    single_dirs = []
    # get model from tasks
    model_dir = config['model_root'] + 'task_' + args.task
    for dom in cross_dom:
        dom_name = config['cross_domain_targets'][args.task][dom]
        single_dirs.append(f"{model_dir}/{dom_name}_BPR.pt")
        if args.target_domain == dom:
            args.target_dirs = single_dirs[-1]
    args.single_dirs = single_dirs
    args.model_dir = model_dir
    args.model_dir = model_dir
    for val in args.single_dirs:
        print(val)
    print(" ====================================== ")
    print(f" **** cross model training on task {args.task}; Train a new cross model? {args.pre_train}")
    print(" =============== running ============== ")
    main(print_p=True)
