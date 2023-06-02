import os
os.chdir("/home/toonify/stylegan2")

import sys

sys.path.append("/home/toonify/stylegan2")

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    print(gpus)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import pretrained_networks


blended_url = "/home/toonify/ffhq-cartoon-blended-64.pkl"  #download here! https://drive.google.com/uc?id=1H73TfV5gQ9ot7slSed_l-lim9X7pMRiU 
ffhq_url = "/home/toonify/stylegan2-ffhq-config-f.pkl"     #download here! http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl


_, _, Gs_blended = pretrained_networks.load_networks(blended_url)
_, _, Gs = pretrained_networks.load_networks(ffhq_url)

