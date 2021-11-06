import importlib
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from utils.evaluations import save_results
from sklearn.ensemble import IsolationForest as IF
from IPython import embed
from collections import defaultdict
from pandas import DataFrame
from sklearn.svm import OneClassSVM

def normalize(imgs):
    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])
    mean = mean[np.newaxis, :, np.newaxis, np.newaxis]
    std = std[np.newaxis, :, np.newaxis, np.newaxis]
    return (imgs - mean)/std

def sim(m1, m2):
    m1_ = m1/m1.norm(dim = 1, keepdim = True)
    m2_ = m2/m2.norm(dim = 1, keepdim = True)
    sim_mat = torch.mm(m1_, m2_.t())
    return torch.norm(sim_mat)

def train_and_test(args):
    src_name = args.src
    tgt_name = args.tgt
    label = args.label
    batch_size = args.batch_size
    latent_dim = args.l_dim
    nb_epochs = args.nb_epochs
    lr = args.lr

    size = (args.size, args.size)
    src_data = importlib.import_module("data.{}".format(src_name))

    tgt_data = importlib.import_module("data.{}".format(tgt_name))

    src_X_train = src_data.source_load(label, size)
    tgt_X_train, X_test, y_test = tgt_data.target_load(label, args.tgt_num, size)
    if tgt_name == "svhn":
        tgt_X_train = tgt_X_train.mean(axis = 1, keepdims = True)
        X_test = X_test.mean(axis = 1, keepdims = True)
    if src_name == "svhn":
        src_X_train = src_X_train.mean(axis = 1, keepdims = True)

    if src_name in ["clipart", "product", "wood", "carpet", "leather"]:
        print ("Normalzie the input as required by Pytorch.")
        src_X_train, tgt_X_train, X_test = normalize(src_X_train), normalize(tgt_X_train), normalize(X_test)
    else:
        src_X_train, tgt_X_train, X_test = 2*src_X_train - 1, 2*tgt_X_train - 1, 2*X_test - 1

    net_tgt = importlib.import_module('IRAD.{}_utils'.format(tgt_name))
    net_src = importlib.import_module('IRAD.{}_utils'.format(src_name))

    num_train = src_X_train.shape[0]

    # make the size of tgt_X_train the same as src_X_train
    indices = np.random.choice(tgt_X_train.shape[0], num_train)

    tgt_X_train = tgt_X_train[indices, :]

    num_batches = math.ceil(num_train/batch_size)
    print ("Number of batch", num_batches)
    Enc_src = net_src.Encoder
    Dec_src = net_src.Decoder
    Dis_src = net_src.Discriminator
    Dis_z = net_src.Dis_z

    if args.gan_loss == "gan":
        crit = nn.BCELoss()
        print ("Using traditional GAN")
    elif args.gan_loss == "lsgan":
        crit = torch.nn.MSELoss()
        print ("Using least square GAN")
    mse_loss = nn.MSELoss()

    gen_src = Dec_src(latent_dim)
    enc_src = Enc_src(latent_dim)
    enc_tgt = Enc_src(latent_dim)
    enc_share = Enc_src(latent_dim)
    dis_src = Dis_src()
    dis_z = Dis_z(latent_dim)

    gen_src   = gen_src.cuda()
    enc_src   = enc_src.cuda()
    enc_tgt   = enc_tgt.cuda()
    enc_share = enc_share.cuda()
    dis_src   = dis_src.cuda()
    dis_z     = dis_z.cuda()
    crit     = crit.cuda()

    optim_gen_src   = optim.Adam(gen_src.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_enc_src   = optim.Adam(enc_src.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_enc_share = optim.Adam(enc_share.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_dis_src   = optim.Adam(dis_src.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_dis_z     = optim.Adam(dis_z.parameters(), lr=lr, betas=(0.5, 0.999))

    tmp = []
    for epoch in range(nb_epochs):

        enc_share.train()
        gen_src.train()
        dis_src.train()
        dis_z.train()

        t1 = time.time()
        gen_src.train()
        enc_src.train()
        enc_tgt.train()
        enc_share.train()
        dis_src.train()
        dis_z.train()

        for i in range(num_batches):
            id_start = i*batch_size
            id_end = (i+1)*batch_size
            x_src = torch.FloatTensor(src_X_train[id_start:id_end, :]).cuda()
            x_tgt = torch.FloatTensor(tgt_X_train[id_start:id_end, :]).cuda()
            num_examples = x_src.size()[0]
            true_labels = torch.full((num_examples, 1), 1.0).cuda()
            fake_labels = torch.full((num_examples, 1), 0.0).cuda()

            z_src = enc_src(x_src)
            z_tgt = enc_tgt(x_tgt)
            z_share_src = enc_share(x_src)
            z_share_tgt = enc_share(x_tgt)
            z_rand = torch.randn([num_examples, latent_dim]).cuda()

            x_src_recon        = gen_src(z_src + z_share_src)
            x_src_recon_fusion = gen_src(z_src + z_share_tgt)
            x_src_fake         = gen_src(z_rand + z_share_src)

            loss_cycle_1 = mse_loss(x_src, x_src_recon)
            loss_cycle_2 = mse_loss(x_src, x_src_recon_fusion)



            logits_src = dis_src(x_src)
            logits_src_recon = dis_src(x_src_recon)
            logits_src_recon_fusion = dis_src(x_src_recon_fusion)

            logits_src_fake = dis_src(x_src_fake)

            logits_z_src = dis_z(z_share_src)
            logits_z_tgt = dis_z(z_share_tgt)

            loss_sim_src = sim(z_share_src, z_src)
            loss_sim_src = torch.zeros([1,1]).cuda()

            loss_sim_share = -sim(z_share_tgt, z_share_src)
            loss_sim_share = torch.zeros([1,1]).cuda()


            loss_dis_src = crit(logits_src, true_labels) + crit(logits_src_recon, fake_labels)\
                         + crit(logits_src_recon_fusion, fake_labels)\
                         + crit(logits_src_fake, fake_labels)
            dis_src.zero_grad()
            loss_dis_src.backward(retain_graph=True)
            optim_dis_src.step()

            loss_dis_z = crit(logits_z_src, true_labels) + crit(logits_z_tgt, fake_labels)
            dis_z.zero_grad()
            loss_dis_z.backward(retain_graph=True)
            optim_dis_z.step()

            loss_enc_src = crit(logits_src_recon, true_labels) + crit(logits_src_recon_fusion, true_labels) + loss_cycle_1 + loss_cycle_2 + loss_sim_src
            enc_src.zero_grad()
            loss_enc_src.backward(retain_graph=True)
            optim_enc_src.step()


            loss_enc_share = crit(logits_src_recon, true_labels) + crit(logits_src_recon_fusion, true_labels)\
                           + crit(logits_src_fake, true_labels)\
                           + loss_cycle_1 + loss_cycle_2 + loss_sim_src + loss_sim_share\
                           + crit(logits_z_src, fake_labels) + crit(logits_z_tgt, true_labels)

            enc_share.zero_grad()
            loss_enc_share.backward(retain_graph=True)
            optim_enc_share.step()

            loss_gen_src = crit(logits_src_recon, true_labels) + crit(logits_src_recon_fusion, true_labels)\
                         + crit(logits_src_fake, true_labels)\
                         + loss_cycle_1 + loss_cycle_2
            gen_src.zero_grad()
            loss_gen_src.backward()
            optim_gen_src.step()

        print("Epoch %d | time = %ds | loss gen_src = %.4f | loss enc_share = %.4f | loss dis_share = %.4f | sim share = %f"
              % (epoch, time.time() - t1, loss_gen_src.item(), loss_enc_share.item(), loss_dis_src.item(), loss_sim_share.item()))
    tmp.append(test(enc_share, gen_src, dis_src, dis_z, src_X_train, tgt_X_train, X_test, y_test, args))
    return enc_share, enc_src, enc_tgt, gen_src, dis_src, dis_z, src_X_train, tgt_X_train, X_test, y_test, tmp

def test(enc_share, gen_src, dis_src, dis_z, src_X_train, tgt_X_train, X_test, y_test, args):
    enc_share.eval()
    gen_src.eval()

    dis_src.eval()
    dis_z.eval()

    random_seed = args.rd
    label = args.label
    dataset = args.tgt
    batch_size = 6
    print ("Fix testing batch size as 6!")
    num_test = X_test.shape[0]
    batch_test = math.ceil(num_test/batch_size)

    x_normal_all = np.concatenate([src_X_train, tgt_X_train], axis = 0)
    num_normal = x_normal_all.shape[0]
    batch_normal = math.ceil(num_normal/batch_size)
    encoded_normal_all = []
    for i in range(batch_normal):
        id_start = i*batch_size
        id_end = (i + 1)*batch_size
        x = torch.FloatTensor(x_normal_all[id_start:id_end, :]).cuda()
        encoded = enc_share(x)
        encoded = encoded.cpu().detach().numpy()
        encoded_normal_all.append(encoded)
    encoded_normal_all = np.concatenate(encoded_normal_all, axis = 0)
    if args.ad_model == "if":
        ad_model = IF(n_estimators=100, contamination = 0)
    elif args.ad_model == "ocsvm":
        ad_model = OneClassSVM(gamma="auto", kernel="poly")
    ad_models = [IF(n_estimators=100, contamination = 0), OneClassSVM(gamma="auto", kernel="poly")]
    for ad_model in ad_models:
        ad_model.fit(encoded_normal_all)

    encoded_test_all = []
    for i in range(batch_test):
        id_start = i*batch_size
        id_end = (i + 1)*batch_size
        x = torch.FloatTensor(X_test[id_start:id_end, :]).cuda()
        encoded = enc_share(x)
        encoded = encoded.cpu().detach().numpy()
        if len(encoded.shape) == 1:
            encoded = encoded[np.newaxis, ...]
        encoded_test_all.append(encoded)

    encoded_test_all = np.concatenate(encoded_test_all, axis = 0)

    scores = [-ad_model.score_samples(encoded_test_all) for ad_model in ad_models]
    print("IF")
    rst1 = save_results(scores[0], y_test, 'IRAD', dataset, "IRAD", 0, label, random_seed)
    print("OCSVM")
    rst2 = save_results(scores[1], y_test, 'IRAD', dataset, "IRAD", 0, label, random_seed)
    if args.ad_model == "if":
        return rst1
    elif args.ad_model == "ocsvm":
        return rst2

def run(args):
    enc_share, enc_src, enc_tgt, gen_src, dis_src, dis_z, src_X_train, tgt_X_train, X_test, y_test, tmp = train_and_test(args)
