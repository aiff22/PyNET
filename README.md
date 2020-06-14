## Replacing Mobile Camera ISP with a Single Deep Learning Model

<br/>

<img src="http://people.ee.ethz.ch/~ihnatova/assets/img/pynet/pynet_teaser.jpg"/>

<br/>

#### 1. Overview [[Paper]](https://arxiv.org/pdf/2002.05509.pdf) [[PyTorch Implementation]](https://github.com/aiff22/PyNET-PyTorch) [[Project Webpage]](http://people.ee.ethz.ch/~ihnatova/pynet.html)

This repository provides the implementation of the RAW-to-RGB mapping approach and PyNET CNN presented in [this paper](https://arxiv.org/). The model is trained to convert **RAW Bayer data** obtained directly from mobile camera sensor into photos captured with a professional Canon 5D DSLR camera, thus replacing the entire hand-crafted ISP camera pipeline. The provided pre-trained PyNET model can be used to generate full-resolution **12MP photos** from RAW (DNG) image files captured using the Sony Exmor IMX380 camera sensor. More visual results of this approach for the Huawei P20 and BlackBerry KeyOne smartphones can be found [here](http://people.ee.ethz.ch/~ihnatova/pynet.html#demo).

<br/>

#### 2. Prerequisites

- Python: scipy, numpy, imageio and pillow packages
- [TensorFlow 1.X](https://www.tensorflow.org/install/) + [CUDA cuDNN](https://developer.nvidia.com/cudnn)
- Nvidia GPU

<br/>

#### 3. First steps

- Download the pre-trained [VGG-19 model](https://polybox.ethz.ch/index.php/s/7z5bHNg5r5a0g7k) <sup>[Mirror](https://drive.google.com/file/d/0BwOLOmqkYj-jMGRwaUR2UjhSNDQ/view?usp=sharing)</sup> and put it into `vgg_pretrained/` folder.
- Download the pre-trained [PyNET model](https://drive.google.com/file/d/1txsJaCbeC-Tk53TPlvVk3IPpRw1Ro3BS/view?usp=sharing) and put it into `models/original/` folder.
- Download [Zurich RAW to RGB mapping dataset](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset) and extract it into `raw_images/` folder.    
  <sub>This folder should contain three subfolders: `train/`, `test/` and `full_resolution/`</sub>
  
  <sub>*Please note that Google Drive has a quota limiting the number of downloads per day. To avoid it, you can login to your Google account and press "Add to My Drive" button instead of a direct download. Please check [this issue](https://github.com/aiff22/PyNET/issues/4) for more information.* </sub>

<br/>


#### 4. PyNET CNN

<br/>

<img src="http://people.ee.ethz.ch/~ihnatova/assets/img/pynet/pynet_architecture_github.png" alt="drawing" width="1000"/>

<br/>

PyNET architecture has an inverted pyramidal shape and is processing the images at **five different scales** (levels). The model is trained sequentially, starting from the lowest 5th layer, which allows to achieve good reconstruction results at smaller image resolutions. After the bottom layer is pre-trained, the same procedure is applied to the next level till the training is done on the original resolution. Since each higher level is getting **upscaled high-quality features** from the lower part of the model, it mainly learns to reconstruct the missing low-level details and refines the results. In this work, we are additionally using one transposed convolutional layer (Level 0) on top of the model that upsamples the image to its target size.

<br/>

#### 5. Training the model

The model is trained level by level, starting from the lowest (5th) one:

```bash
python train_model.py level=<level>
```

Obligatory parameters:

>```level```: **```5, 4, 3, 2, 1, 0```**

Optional parameters and their default values:

>```batch_size```: **```50```** &nbsp; - &nbsp; batch size [small values can lead to unstable training] <br/>
>```train_size```: **```30000```** &nbsp; - &nbsp; the number of training patches randomly loaded each 1000 iterations <br/>
>```eval_step```: **```1000```** &nbsp; - &nbsp; each ```eval_step``` iterations the accuracy is computed and the model is saved <br/>
>```learning_rate```: **```5e-5```** &nbsp; - &nbsp; learning rate <br/>
>```restore_iter```: **```None```** &nbsp; - &nbsp; iteration to restore (when not specified, the last saved model for PyNET's ```level+1``` is loaded)<br/>
>```num_train_iters```: **```5K, 5K, 20K, 20K, 35K, 100K (for levels 5 - 0)```** &nbsp; - &nbsp; the number of training iterations <br/>
>```vgg_dir```: **```vgg_pretrained/imagenet-vgg-verydeep-19.mat```** &nbsp; - &nbsp; path to the pre-trained VGG-19 network <br/>
>```dataset_dir```: **```raw_images/```** &nbsp; - &nbsp; path to the folder with **Zurich RAW to RGB dataset** <br/>

</br>

Below we provide the commands used for training the model on the Nvidia Tesla V100 GPU with 16GB of RAM. When using GPUs with smaller amount of memory, the batch size and the number of training iterations should be adjusted accordingly:

```bash
python train_model.py level=5 batch_size=50 num_train_iters=5000
python train_model.py level=4 batch_size=50 num_train_iters=5000
python train_model.py level=3 batch_size=48 num_train_iters=20000
python train_model.py level=2 batch_size=18 num_train_iters=20000
python train_model.py level=1 batch_size=12 num_train_iters=35000
python train_model.py level=0 batch_size=10 num_train_iters=100000
```

<br/>

#### 6. Test the provided pre-trained models on full-resolution RAW image files

```bash
python test_model.py level=0 orig=true
```

Optional parameters:

>```use_gpu```: **```true```**,**```false```** &nbsp; - &nbsp; run the model on GPU or CPU <br/>
>```dataset_dir```: **```raw_images/```** &nbsp; - &nbsp; path to the folder with **Zurich RAW to RGB dataset** <br/>

<br/>

#### 7. Test the obtained model on full-resolution RAW image files

```bash
python test_model.py level=<level>
```

Obligatory parameters:

>```level```: **```5, 4, 3, 2, 1, 0```**

Optional parameters:

>```restore_iter```: **```None```** &nbsp; - &nbsp; iteration to restore (when not specified, the last saved model for level=```<level>``` is loaded)<br/>
>```use_gpu```: **```true```**,**```false```** &nbsp; - &nbsp; run the model on GPU or CPU <br/>
>```dataset_dir```: **```raw_images/```** &nbsp; - &nbsp; path to the folder with **Zurich RAW to RGB dataset** <br/>

<br/>

#### 8. Folder structure

>```models/```            &nbsp; - &nbsp; logs and models that are saved during the training process <br/>
>```models/original/```   &nbsp; - &nbsp; the folder with the provided pre-trained PyNET model <br/>
>```raw_images/```        &nbsp; - &nbsp; the folder with Zurich RAW to RGB dataset <br/>
>```results/```           &nbsp; - &nbsp; visual results for small image patches that are saved while training <br/>
>```results/full-resolution/``` &nbsp; - &nbsp; visual results for full-resolution RAW image data saved during the testing <br/>
>```vgg-pretrained/```    &nbsp; - &nbsp; the folder with the pre-trained VGG-19 network <br/>

>```load_dataset.py```    &nbsp; - &nbsp; python script that loads training data <br/>
>```model.py```           &nbsp; - &nbsp; PyNET implementation (TensorFlow) <br/>
>```train_model.py```     &nbsp; - &nbsp; implementation of the training procedure <br/>
>```test_model.py```      &nbsp; - &nbsp; applying the pre-trained model to full-resolution test images <br/>
>```utils.py```           &nbsp; - &nbsp; auxiliary functions <br/>
>```vgg.py```             &nbsp; - &nbsp; loading the pre-trained vgg-19 network <br/>

<br/>

#### 9. Bonus files

These files can be useful for further experiments with the model / dataset:

>```dng_to_png.py```            &nbsp; - &nbsp; convert raw DNG camera files to PyNET's input format <br/>
>```evaluate_accuracy.py```     &nbsp; - &nbsp; compute PSNR and MS-SSIM scores on Zurich RAW-to-RGB dataset for your own model <br/>

<br/>

#### 10. License

Copyright (C) 2020 Andrey Ignatov. All rights reserved.

Licensed under the [CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

The code is released for academic research use only.

<br/>

#### 11. Citation

```
@article{ignatov2020replacing,
  title={Replacing Mobile Camera ISP with a Single Deep Learning Model},
  author={Ignatov, Andrey and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2002.05509},
  year={2020}
}
```
<br/>

#### 12. Any further questions?

```
Please contact Andrey Ignatov (andrey@vision.ee.ethz.ch) for more information
```
