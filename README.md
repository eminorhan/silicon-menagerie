## Wilde beasts of silicon

> *... come ye, assemble all the beasts of the field, come to devour ...* &ndash;Jeremiah 12:9

This is a stand-alone repository to facilitate the use of all models I have trained on SAYCam (and more!). It is still in progress. More models and further functionalities will be forthcoming shortly. The models are all hosted on [Huggingface](https://huggingface.co/eminorhan), which, to my not inconsiderable astonishment, seems to offer free unlimited storage for models and datasets (thanks Huggingface!).

### Image embedding models

Model names are specified in the format `x_y_z`, where `x` is the SSL algorithm used to train the model (`dino`, `mugs`, or `mae`), `y` is the data used for training the model (`say`, `s`, `a`, `y`, `imagenet_100`, `imagenet_10`, `imagenet_3`, or `imagenet_1`), and `z` is the model architecture (`resnext50`, `vitb14`, `vitl16`, `vitb16`, `vits16`). Please note that not all possible combinations are available at this time (see [here](https://huggingface.co/eminorhan) for a list of all available models). You will get an error if you try to load an unavailable model. 

Loading a pretrained model is then as easy as:

```python
from utils import load_model

model = load_model('dino_s_vitb14')
```

This will download the corresponding pretrained checkpoint, store it in cache, build the right model architecture, and load the pretrained weights onto the model, all in one go! When you load a model, you will get a warning message that says something like `_IncompatibleKeys(missing_keys=[], unexpected_keys=...)`. That's OK, don't panic! Life is like that sometimes. This is because we're not loading the projection head used during DINO or Mugs pretraining, or the decoder used during MAE pretraining. We're only interested in the encoder backbone.

### Generative image models

These are generative models that can be used to generate images. For these models, we first learn a discrete codebook of size 8192 with a [VQ-GAN](https://github.com/CompVis/taming-transformers) model and then encode the video frames as 32x32 integers from this codebook. These discretized and spatially downsampled frames are then fed into a GPT model to learn a prior over the frames. The two parts of the model are shared separately below. The `encoder-decoder` part can be used to encode images with the discrete codebook, as well as decode images (to 256x256 pixels) given a discrete latent representation. The `GPT` part can be used to generate (or sample) new discrete latent representations.

TBD

### Generative video models

TBD
