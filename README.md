## Model zoo for SAYCam

This is a stand-alone repository containing all models I have trained on SAYCam thus far. It is still in progress. More models and details on how to load these models will be forthcoming shortly.

### Image embedding models

The models below were trained with a variety of self-supervised learning algorithms and can be used to generate image embeddings (*i.e.* these are non-generative models). Training logs contain useful information about the training arguments and the training losses. 

#### Mugs

[Mugs](https://github.com/sail-sg/mugs) is a multi-granular self-supervised learning algorithm. We trained ViT models of different sizes with the Mugs algorithm. 

ViT-L/16:

* Mugs SAY ViT-L/16: [checkpoint](https://drive.google.com/file/d/1-vMZHxnBTbduhLytDQq1q3ozJRIBY230/view?usp=sharing), [training log](https://github.com/eminorhan/saycam-zoo/blob/master/pretrain_logs/mugs/say_5fps_vitl16_log.txt)

* Mugs S ViT-L/16: [checkpoint](https://drive.google.com/file/d/17yryncnrw1-ZERd00kDDf0atCU_SpsRl/view?usp=sharing), [training log](https://github.com/eminorhan/saycam-zoo/blob/master/pretrain_logs/mugs/s_5fps_vitl16_log.txt)

* Mugs A ViT-L/16: [checkpoint](https://drive.google.com/file/d/1mlqMzytofMe69wgCkQ__5G22EUrMwT-8/view?usp=sharing), [training log](https://github.com/eminorhan/saycam-zoo/blob/master/pretrain_logs/mugs/a_5fps_vitl16_log.txt)

* Mugs Y ViT-L/16: [checkpoint](https://drive.google.com/file/d/1nzzxLarpy93Y7vTYXsIVVkl4A-tr1WQ-/view?usp=sharing), [training log](https://github.com/eminorhan/saycam-zoo/blob/master/pretrain_logs/mugs/y_5fps_vitl16_log.txt)

ViT-B/16:

* Mugs SAY ViT-B/16: [checkpoint](https://drive.google.com/file/d/1Fw9auROFdumEpNc--bIEpu-9rAOWsewt/view?usp=sharing), [training log](https://github.com/eminorhan/saycam-zoo/blob/master/pretrain_logs/mugs/say_5fps_vitb16_log.txt)

* Mugs S ViT-B/16: [checkpoint](https://drive.google.com/file/d/1GEkUXg0Rtkii-A-KZlCEhCNGf87Xg74C/view?usp=sharing), [training log](https://github.com/eminorhan/saycam-zoo/blob/master/pretrain_logs/mugs/s_5fps_vitb16_log.txt)

* Mugs A ViT-B/16: [checkpoint](https://drive.google.com/file/d/1BAZfzR9wYXTKTJ3H9kjtawfZJtkR1NoW/view?usp=sharing), [training log](https://github.com/eminorhan/saycam-zoo/blob/master/pretrain_logs/mugs/a_5fps_vitb16_log.txt)

* Mugs Y ViT-B/16: [checkpoint](https://drive.google.com/file/d/1aUlsChRfqxu-JrZpHmLYrksi-ek8m_TM/view?usp=sharing), [training log](https://github.com/eminorhan/saycam-zoo/blob/master/pretrain_logs/mugs/y_5fps_vitb16_log.txt)

ViT-S/16:

* Mugs SAY ViT-S/16: [checkpoint](https://drive.google.com/file/d/1D8kY3T1uixHflaQ_fJCJP4zH_kimq5yH/view?usp=sharing), [training log](https://github.com/eminorhan/saycam-zoo/blob/master/pretrain_logs/mugs/say_5fps_vits16_log.txt)

* Mugs S ViT-S/16: [checkpoint](https://drive.google.com/file/d/1SEZyCMKDBoH4snV1jJrAQL7p8w2GUU69/view?usp=sharing), [training log](https://github.com/eminorhan/saycam-zoo/blob/master/pretrain_logs/mugs/s_5fps_vits16_log.txt)

* Mugs A ViT-S/16: [checkpoint](https://drive.google.com/file/d/1f9m2dlbStw7eEr_IGUAOWgaNiPPVQVgu/view?usp=sharing), [training log](https://github.com/eminorhan/saycam-zoo/blob/master/pretrain_logs/mugs/a_5fps_vits16_log.txt)

* Mugs Y ViT-S/16: [checkpoint](https://drive.google.com/file/d/12KhcM52Up0Yw2hFft5GNysOXd5x5pzte/view?usp=sharing), [training log](https://github.com/eminorhan/saycam-zoo/blob/master/pretrain_logs/mugs/y_5fps_vits16_log.txt)

#### DINO

[DINO](https://github.com/facebookresearch/dino) is a self-distillation based self-supervised learning algorithm. We trained ViT and ResNeXt models of different sizes with the DINO algorithm.

ViT-L/8:

* DINO SAY ViT-L/8: [checkpoint](), [training log]()

* DINO S ViT-L/8: [checkpoint](), [training log]()

* DINO A ViT-L/8: [checkpoint](), [training log]()

* DINO Y ViT-L/8: [checkpoint](), [training log]()

ViT-L/16:

* DINO SAY ViT-L/16: [checkpoint](), [training log]()

* DINO S ViT-L/16: [checkpoint](), [training log]()

* DINO A ViT-L/16: [checkpoint](), [training log]()

* DINO Y ViT-L/16: [checkpoint](), [training log]()

ViT-B/16:

* DINO SAY ViT-B/16: [checkpoint](), [training log]()

* DINO S ViT-B/16: [checkpoint](), [training log]()

* DINO A ViT-B/16: [checkpoint](), [training log]()

* DINO Y ViT-B/16: [checkpoint](), [training log]()

ViT-S/16:

* DINO SAY ViT-S/16: [checkpoint](), [training log]()

* DINO S ViT-S/16: [checkpoint](), [training log]()

* DINO A ViT-S/16: [checkpoint](), [training log]()

* DINO Y ViT-S/16: [checkpoint](), [training log]()

ResNeXt-101_32x8d:

* DINO SAY ResNeXt-101_32x8d: [checkpoint](), [training log]()

* DINO S ResNeXt-101_32x8d: [checkpoint](), [training log]()

* DINO A ResNeXt-101_32x8d: [checkpoint](), [training log]()

* DINO Y ResNeXt-101_32x8d: [checkpoint](), [training log]()

ResNeXt-50_32x4d:

* DINO SAY ResNeXt-50_32x4d: [checkpoint](), [training log]()

* DINO S ResNeXt-50_32x4d: [checkpoint](), [training log]()

* DINO A ResNeXt-50_32x4d: [checkpoint](), [training log]()

* DINO Y ResNeXt-50_32x4d: [checkpoint](), [training log]()

### Generative image models

These are generative models that can be used to generate images.

* VQ-GAN-GPT SAY:

### Generative video models

These are generative models that can be used to generate video clips.

* VQ-VAE-GPT:

