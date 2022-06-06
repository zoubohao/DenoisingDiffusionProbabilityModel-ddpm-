# DenoisingDiffusionProbabilityModel
This may be the simplest implement of DDPM. <br>
**HOW TO RUN**
* 1.  You can run Main.py to train the UNet on CIFAR-10 dataset. After training, you can set the parameters in the model config to see the amazing process of DDPM.
* 2.  You can run MainCondition.py to train UNet on CIFAR-10. This is for DDPM + Classifier free guidence.

Some generated images are showed below:

![Generated Images without condition](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/SampledImgs/sampled_80_noCond.png)

![Generated Images with condition](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/SampledImgs/sampled_80_63.png)
