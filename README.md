# Get-an-education-for-Computer-Vision
๐ Complete In-depth course in Computer Vision

[์์ธICT์ด๋ธ๋ฒ ์ด์์คํ์ด](https://ict.eksa.or.kr/portal/applyconfirm_ict/main.user?paramMap.finalGbn=N)์์ ๊ต์ก์์ ์ ๋ฐํ์ฌ ์งํํ๋ ์ธ๊ณต์ง๋ฅ ์ฌํ ์๊ฐ์ง๋ฅ ํ๋ก์ ํธ ๊ฐ๋ฐ ๊ณผ์ ์ ์๋ฃํจ.
> 2021.07.05 ~ 2021.08.30 ๊ธฐ๊ฐ๋์ ์ด 160์๊ฐ์ ๊ต์ก๊ณผ์  ์๋ฃ   
> Computer Vision์ ํ๋ฆ๊ณผ ์ต์  ๊ธฐ์  ๋ฐ ๋ํฅ์ ๊ณต๋ถ   
> __๊ฐ์ธ์ ์ผ๋ก ํฌ๊ฒ ์ฑ์ฅํ  ์ ์๋ ๊ธฐํ์์ผ๋ฉฐ, CV๋ถ์ผ์ ๋น ์ ธ ์ฆ๊ฒ๊ฒ ๊ณต๋ถํ๋ ๊ฒฝํ์๋๋ค.__   
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## week1 - Image Classification
+ ๊ธฐ๋ณธ์ ์ธ NN์ ์์ ํ/์ญ์ ํ
+ transfer learning & Data Augmentation
+ DenseNet - [Densely Connected Convolutional Networks(2017)](https://arxiv.org/pdf/1608.06993.pdf) ๋ผ๋ฌธ๋ฆฌ๋ทฐ
+ vision transformer - [AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE(2021)](https://arxiv.org/pdf/2010.11929.pdf) ๋ผ๋ฌธ๋ฆฌ๋ทฐ
+ VIT ์ฝ๋ ๋ฆฌ๋ทฐ [code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/Image%20Classification/image_classification_with_vision_transformer.ipynb)
#### ๊ณผ์ 1. Classification Project - Kaggle์ ์๋ classification task๋ฅผ ์งํํ์ฌ ์ธ์ฌ์ดํธ ๋ฝ์๋ณด๊ธฐ.
[๊ณผ์ 1์ ์ํํ ๊ฐ์ธ ๋ธ๋ก๊ทธ ํฌ์คํ](https://inhovation97.tistory.com/43)   
[DenseNet๋ผ๋ฌธ์ ๋ฆฌ๋ทฐํ ๊ฐ์ธ ๋ธ๋ก๊ทธ ํฌ์คํ](https://inhovation97.tistory.com/47)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week2 - Image Segmentation
+ Segmentation U-net ๊ฐ๋ & ๋ชจ๋ธ ๊ตฌํ [code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/Image%20Segmentation/Unet_tutorial_origin.ipynb)
+ Dense Block with U-net ์ค์ต [code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/Image%20Segmentation/DensUnet_tutorial_ImgGenerator_Attention_210715.ipynb)
+ SENet - [Squeeze and Excitation Networks(attention)(2018)](https://arxiv.org/pdf/1709.01507) ๋ผ๋ฌธ ๋ฆฌ๋ทฐ
+ PSPNet - [Pyramid Scene Parsing Network(2016)](https://arxiv.org/pdf/1612.01105.pdf) ๋ผ๋ฌธ ๋ฆฌ๋ทฐ
#### ๊ณผ์ 2. Segmentation Project - attention block & loss function์ ์ด์ฉํ์ฌ ์ฑ๋ฅ์ ํฅ์ํ๊ธฐ.
[code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/Image%20Segmentation/ResUnet_tutorial_%EB%8B%A4%EC%A4%91%EB%B6%84%EB%A5%98softmax_with_attention.ipynb)
> ๊ธฐ๋ณธ์ ์ธ U-Net๊ณผ attention ๋ชจ๋์ ์ถ๊ฐํ residual U-Net์ ์ฑ๋ฅ ์ฐจ์ด๋ฅผ ๋น๊ต.   
>    
> ![image](https://user-images.githubusercontent.com/59557720/161203850-05dfbd40-8d0e-4c1c-8244-056b21a57242.png)   
> ๋ ๋ชจ๋ธ ๋ชจ๋ epoch 20 ์์ค์์์ ์ฑ๋ฅ ๊ฒฐ๊ณผ์ด๋ฉฐ, ๋น๊ต์  segmentationํ๊ธฐ ์ฌ์ด ์ด๋ฏธ์ง๋ก ๋น๊ตํจ   
> -> ์ก์์ ์ผ๋ก๋ ์์ ๊ฐ์ ํฐ ์ฐจ์ด๊ฐ ๋ฌ์.   
>    
> ์ค๋ฅธ์ชฝ ๋ชจ๋ธ์ loss function์ dice_loss๋ฅผ ์ด์ฉํ์ผ๋ฉฐ, binary๊ฐ ์๋ categorical CE๋ฅผ ์ด์ฉํจ   
> -> ๋ถ๋ฅํ๊ธฐ ํ๋  ๊ฒฝ๊ณ๋ฉด ๋ถ๋ฅ๊ฐ ์ํ๋์ด ๋ ๋์ ์ฑ๋ฅ์ ๊ธฐ๋   


[Squeeze and Excitation Networks(attention)๋ผ๋ฌธ์ ๋ฆฌ๋ทฐํ ๊ฐ์ธ ๋ธ๋ก๊ทธ](https://inhovation97.tistory.com/48)   
[CBAM(attention)๋ผ๋ฌธ์ ๋ฆฌ๋ทฐํ ๊ฐ์ธ ๋ธ๋ก๊ทธ](https://inhovation97.tistory.com/63)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week 3 - GAN 1
+ DeepLab v3+ - [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation(2018)](https://arxiv.org/pdf/1802.02611.pdf) ๋ผ๋ฌธ ๋ฆฌ๋ทฐ
+ Auto Encoder & Variational Auto Encoder ๊ฐ๋ & ์ค์ต   
+ Denoising VAE ์ค์ต [code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/GAN/vae%EA%B3%BC%EC%A0%9C_denoising_autoencoder_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%89%E1%85%B3%E1%86%B8.ipynb)   
+ DCGAN ๊ตฌํ ์ธ๋ถ ์ฌํญ & ๋ชฉ์  ํจ์
> GAN์ _๋ฏธ์ ๊ด์ GAN ๋ฅ๋ฌ๋ ์ค์  ํ๋ก์ ํธ_ ์์ ์ ๋๋ด๋ ๊ฒ์ ๋ชฉํ๋ก ๊ฐ์๋ฅผ ์งํ   
>   
> ![image](https://user-images.githubusercontent.com/59557720/161194974-41882c69-eed1-4f5a-b5c9-34c0f6e6ad2c.png)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week 4 - GAN 2
+ pix2pix - [Image-to-Image Translation with Conditional Adversarial Networks(2017)](https://arxiv.org/pdf/1611.07004.pdf) ๋ผ๋ฌธ ๋ฆฌ๋ทฐ
+ WGAN ( Wasserstein loss, Lipshitz ์ ์ฝ ) ๊ฐ๋   
+ Cycle GAN ( ์ฌ๋ฌ๊ฐ์ loss๋ฅผ ์ด์ฉ ) ๊ฐ๋ & ๋ผ๋ฌธ ๋ฆฌ๋ทฐ & ์ค์ต   
+ Neural Style Transfer - [Image Style Transfer Using Convolutional Neural Networks(2015)](https://arxiv.org/pdf/1508.06576.pdf) ๋ผ๋ฌธ ๋ฆฌ๋ทฐ & ์ค์ต [code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/GAN/Style_transfer_20210802.ipynb)   
+ SR GAN - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network(2017)](https://arxiv.org/pdf/1609.04802.pdf) ๋ผ๋ฌธ ๋ฆฌ๋ทฐ & ์ค์ต[code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/GAN/SRGAN_20210615.ipynb)   
#### ๊ฐ์ธ ํ๋ก์ ํธ 1 - fine tuning์ผ๋ก style transfer์ style์ ๋์ฑ ๊ฐ๋ ฅํ๊ฒ ์ํ๋ณด๊ธฐ
> vggnet์ fine tuningํ์ฌ ์ฌ์จ ์ด๋ฏธ์ง๋ฅผ ๋ถ๋ฅํ๋ ๋ชจ๋ธ์ ๋ง๋  ๋ค, ํด๋น ๋ชจ๋ธ์ ConV Network์ ์ด์ฉํ์ฌ ์ข ๋ ๊ฐํ ์ฌ์จํ์ ์ด๋ฏธ์ง๋ฅผ ์ ๋ํด๋ดค์ต๋๋ค!   
> [VggNet fine tuning ์ ์ฉ ์ฝ๋](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/GAN/project1/pretraining_style_transfer.ipynb)   
> [style transfer ์ ์ฉ ์ฝ๋](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/GAN/project1/style_transfer_in_pytorch.ipynb)   
>   
> ![image](https://user-images.githubusercontent.com/59557720/161210449-88875252-8fbd-446c-ab45-3e27a79e5024.png)
>   
> ๋ณด๋ค์ํผ fine tuning ํ๋ฉด ํ์คํ style์ด ๊ฐํ๊ฒ ์ํ์ง๋๋ค.   
> ์ฐ์ฐ์น ์๊ฒ content image์ style image ๋ชจ๋ ๋ฒฝ์ ์ก์๊ฐ ๊ฑธ๋ ค์์๋๋ฐ, ๊ฐ์ ๊ฐ์ฒด๋ก ์ธ์๋์ด ๋งํ ๊ทธ๋๋ก ์ก์๊ฐ ์ํ์ ธ ์์ฃผ ๋ง์กฑ์ค๋ฌ์ ์ต๋๋ค.   
>    
> ์ฌ๋ฌ๊ฐ์ ์ด๋ฏธ์ง๋ฅผ ์ถ๋ก ํ๋๋ฐ, ๋จธ๋ฆฌ ์๋ ์ฌ๋์ ์บ๋ฆญํฐ๋ก ์ธ์ํ์ง ๋ชปํ ๋ฐ๋ฉด ๋จธ๋ฆฌ๊ฐ ์๋ ์ฃผํธ๋ฏผ์ ์บ๋ฆญํฐ๋ก ์ธ์ํ์ฌ ๋ธ๋์์ด ์ํ์ง!   
>    
> ์คํ๋ ค finetuning ๊ณผ์ ์์ ์ฌ์จ ํ์ ์ด๋ฏธ์ง๋ฅผ ๊ณผ์ ํฉ์ํค๊ณ  ์ถ์์ง๋ง, ํฐ ๋ฐ์ดํฐ์์ด๋ ๊ฐ๋ฐ ํ๊ฒฝ ๋ฑ์ด ๋๋ฌด ์์ฌ์ ์.
> 


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week 5 - Object Detection
+ R-CNN -> SPP-Net -> Fast R-CNN -> Faster R-CNN(RPN)
+ YOLO -> SSD -> Retina Net 
+ ์ ์์๋๋ก  One-stage detector์ Two-stage detector๋ฅผ ๋ฐ์  ๊ณ๋ณด ์์๋๋ก ๊ณต๋ถํจ.
+ FPN - [Feature Pyramid Networks for Object Detection(2017)](https://arxiv.org/pdf/1612.03144)
+ YOLO v5 ์ค์ต [code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/Object_detection/train_yolov5_pistols_dataset.ipynb)   

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week 6 - ์ถ๊ฐ ์์ฒญ ๋ผ๋ฌธ ๋ฆฌ๋ทฐ & ๊ฐ์ธ ํ๋ก์ ํธ 2
+ Real-ESRGAN - [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data(2021)](https://arxiv.org/pdf/2107.10833) ๋ผ๋ฌธ๋ฆฌ๋ทฐ   
+ Focal Loss - [Focal Loss for Dense Object Detection(2017)](https://arxiv.org/pdf/1708.02002) ๋ผ๋ฌธ๋ฆฌ๋ทฐ   
+ Tesla AI day - ๋ง์นจ 8์๋ง ํ์ฌ๋ผ๊ฐ AI day์์ ์์ฌ์ ์์จ ์ฃผํ ๊ธฐ์ ์ ํํ ๋ฐํ์ฒ๋ผ ์์ธํ oral presentation์ ํ์ฌ ๊ฐ์ฌ๋๊ณผ ํจ๊ป ๋ค์ผ๋ฉฐ ๊ธฐ์ ์ ๋ํ ์ด์ผ๊ธฐ๋ฅผ ๋๋ - vision์ผ๋ก ์ธ์ฝ๋ฉํ visual ์ ๋ณด๋ฅผ LSTM์ผ๋ก ์ํ์ํ๊ฒ ๋ค๊ฐ๊ฐ ์๋ฎฌ๋ ์ด์ map์ ๋ง๋ค์ด ์ ๊ธฐํ์.   

#### ๊ฐ์ธ ํ๋ก์ ํธ 2 - YOLOv5 ๋ชจ๋ธ์ ํ์ฉํ์ฌ ์ฌ๋ ๋์ด ์์ธก ๋ชจ๋ธ์ ๊ตฌํํด๋ณด๊ธฐ.
> [Roboflow](https://roboflow.com/)ํ๋ซํผ์ ํ์ฉํ์ฌ [CelebA (์ฌ๋ ์ผ๊ตด ๋ฐ์ดํฐ์)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) ์ฝ 800์ฌ ์ฅ์ ์ง์  ๋ผ๋ฒจ๋ง ํ ๋ค, ๋ชจ๋ธ์ ํ์ตํ์ฌ ๋์ด๋ฅผ ์์ธกํ๋ pretrained FC Layer๋ฅผ ํตํด 2stage๋ก ์ค์๊ฐ ์์ธก ๋ชจ๋ธ์ ๊ณํ    
>    
> ![image](https://user-images.githubusercontent.com/59557720/161425381-05068a3e-7c7b-43be-b787-07140adb96d3.png)
>    
> ํ์ง๋ง fc layer๋ฅผ ์ํ age ๋ฐ์ดํฐ์์ ํ๋ณดํ๋๋ฐ์ ์คํจ... ์์ฌ์ด๋๋ก ์ด๋ฏธ ๋ฐฐํฌ๋ cv2๋ผ์ด๋ธ๋ฌ๋ฆฌ ๋ด์ ์กด์ฌํ๋ ์์ฃผ ์ค๋๋ caffe๋ก ์ค๊ณํ ๋ชจ๋ธ์ ํ์ฉํ์ฌ, ์ด๋ฏธ์ง ๊ฒฐ๊ณผ๋ฌผ์ ๋์ถํจ.   
>    
> ~~ํ๋ก์ ํธ ๊ณํ์ ์์ด์๋ ๋ฐ์ดํฐ์ ์ด์๋ฅผ ์ผ๋๋ก ์ง์ผํ๋ค๋ ๊ตํ...~~



