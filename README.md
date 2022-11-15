## Silicon menagerie (like a glass menagerie, but sturdier, more resistant to heartbreaks)

This is a stand-alone repository to facilitate the use of all models I have trained on SAYCam (and more!). It is still in progress. More models and further functionalities will be forthcoming shortly.

### Image embedding models

Loading a model is as easy as:

### Generative image models

These are generative models that can be used to generate images. For these models, we first learn a discrete codebook of size 8192 with a [VQ-GAN](https://github.com/CompVis/taming-transformers) model and then encode the video frames as 32x32 integers from this codebook. These discretized and spatially downsampled frames are then fed into a GPT model to learn a prior over the frames. The two parts of the model are shared separately below. The `encoder-decoder` part can be used to encode images with the discrete codebook, as well as decode images (to 256x256 pixels) given a discrete latent representation. The `GPT` part can be used to generate (or sample) new discrete latent representations. 

* VQ-GAN-GPT SAY: [encoder-decoder](), [GPT]()

* VQ-GAN-GPT S: [encoder-decoder](), [GPT]()

* VQ-GAN-GPT A: [encoder-decoder](), [GPT]()

* VQ-GAN-GPT Y: [encoder-decoder](), [GPT]()

### Generative video models

These are generative models that can be used to generate video clips. The approach here is similar to the above. The only difference is that instead of 2d array, here we model 3d tensors (2 spatial dimensions + 1 temporal dimension) instead.

* VQ-VAE-GPT SAY: [encoder-decoder](), [GPT]()

* VQ-VAE-GPT S: [encoder-decoder](), [GPT]()

* VQ-VAE-GPT A: [encoder-decoder](), [GPT]()

* VQ-VAE-GPT Y: [encoder-decoder](), [GPT]()
