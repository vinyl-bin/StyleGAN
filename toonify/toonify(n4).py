import os
os.chdir("/home/toonify/stylegan2")

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