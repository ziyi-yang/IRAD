#! python3

import argparse
import importlib
import logging
import os
import shutil
import urllib3
import zipfile
import time

def run(args):
    mod_name = "{}.run".format(args.model)

    print("Running script at {}".format(mod_name))

    mod = importlib.import_module(mod_name)
    mod.run(args)

def path(d):
    try:
        assert os.path.isdir(d)
        return d
    except Exception as e:
        raise argparse.ArgumentTypeError("Example {} cannot be located.".format(d))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run examples from the DL 2.0 Anomaly Detector.')
    parser.add_argument('--model', nargs="?", type=path, help='the folder name of the example you want to run e.g gan or bigan')
    parser.add_argument('--src', nargs="?", choices=['mnist', 'usps', 'svhn', 'product', 'clipart', 'leather', 'carpet', 'wood', 'g1', 'g2'], help='the name of the source dataset')
    parser.add_argument('--tgt', nargs="?", choices=['mnist', 'usps', 'svhn', 'product', 'clipart', 'leather', 'carpet', 'wood', 'g1', 'g2'], help='the name of the source dataset')
    parser.add_argument('--batch_size', nargs="?", type=int, default=48, help='batch size')
    parser.add_argument('--lr', nargs="?", type=float, default=1e-5, help='learning rate')
    parser.add_argument('--size', nargs="?", type=int, default=32, help='image (re)size')
    parser.add_argument('--nb_epochs', nargs="?", type=int, default=1, help='number of epochs you want to train the dataset on')
    parser.add_argument('--gpu', nargs="?", type=int, default=0, help='which gpu to use')
    parser.add_argument('--l_dim', nargs="?", type=int, default=32, help='latent dimension')
    parser.add_argument('--gan_loss', nargs="?", default='gan',  choices=['gan', 'lsgan'],
                        help='loss function for GAN training')
    parser.add_argument('--label', nargs="?", type=int, default=0, help='anomalous label for the experiment')
    parser.add_argument('--tgt_num', nargs="?", type=int, default=10, help='number of target data')
    parser.add_argument('--m', nargs="?", default='fm',  choices=['cross-e', 'fm'],
                        help='mode/method for discriminator loss')
    parser.add_argument('--mode', nargs="?", default='tgt',  choices=['trans', 'src', 'tgt', 'aug'],
                        help='mode for training in IF')
    parser.add_argument('--ad_model', nargs="?", default='if',  choices=['if', 'ocsvm'],
                        help='mode for training in IF')
    parser.add_argument('--rd', nargs="?", type=int, default=42,  help='random_seed')

    run(parser.parse_args())
