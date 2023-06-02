import sys

sys.path.append("/stylegan2")

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

# use my copy of the blended model to save Doron's download bandwidth
# get the original here https://mega.nz/folder/OtllzJwa#C947mCCdEfMCRTWnDcs4qw
blended_url = "/stylegan2/ffhq-cartoon-blended-64.pkl"
ffhq_url = "/stylegan2/stylegan2-ffhq-config-f.pkl"


_, _, Gs_blended = pretrained_networks.load_networks(blended_url)
_, _, Gs = pretrained_networks.load_networks(ffhq_url)


import numpy as np
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
from pathlib import Path

latent_dir = Path("generated")
latents = latent_dir.glob("*.npy")
for latent_file in latents:
    latent = np.load(latent_file)
    latent = np.expand_dims(latent, axis=0)
    synthesis_kwargs = dict(
        output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False),
        minibatch_size=8,
    )
    images = Gs_blended.components.synthesis.run(
        latent, randomize_noise=False, **synthesis_kwargs
    )
    Image.fromarray(images.transpose((0, 2, 3, 1))[0], "RGB").save(
        latent_file.parent / (f"{latent_file.stem}-toon.jpg")
    )

from IPython.display import Image, display

embedded = Image(filename="generated/image_01.png", width=256)
tooned = Image(filename="generated/image_01-toon.jpg", width=256)
