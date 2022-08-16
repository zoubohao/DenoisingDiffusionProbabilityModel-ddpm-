# DenoisingDiffusionProbabilityModel
This may be the simplest implement of DDPM. I trained with CIFAR-10 dataset. The links of pretrain weight, which trained on CIFAR-10 are in the Issue 2. <br>
<br>
If you really want to know more about the framwork of DDPM, I have listed some papers for reading by order in the closed Issue 1.
<br>
<br>
Lil' Log is also a very nice blog for understanding the details of DDPM, the reference is 
"https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#:~:text=Diffusion%20models%20are%20inspired%20by,data%20samples%20from%20the%20noise."
<br>
<br>
**HOW TO RUN**
* 1.  You can run Main.py to train the UNet on CIFAR-10 dataset. After training, you can set the parameters in the model config to see the amazing process of DDPM.
* 2.  You can run MainCondition.py to train UNet on CIFAR-10. This is for DDPM + Classifier free guidence.

Some generated images are showed below:

* 1. DDPM without guidence:

![Generated Images without condition](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/SampledImgs/SampledNoGuidenceImgs.png)

* 2. DDPM + Classifier free guidence:

![Generated Images with condition](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/SampledImgs/SampledGuidenceImgs.png)
