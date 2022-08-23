# Get-an-education-for-Computer-Vision
📜 Complete In-depth course in Computer Vision

[서울ICT이노베이션스퀘어](https://ict.eksa.or.kr/portal/applyconfirm_ict/main.user?paramMap.finalGbn=N)에서 교육생을 선발하여 진행하는 인공지능 심화 시각지능 프로젝트 개발 과정을 수료함.
> 2021.07.05 ~ 2021.08.30 기간동안 총 160시간의 교육과정 수료   
> Computer Vision의 흐름과 최신 기술 및 동향을 공부   
> __개인적으로 크게 성장할 수 있는 기회였으며, CV분야를 흥미롭게공부__   
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## week1 - Image Classification
+ 기본적인 NN의 순전파/역전파
+ transfer learning & Data Augmentation
+ DenseNet - [Densely Connected Convolutional Networks(2017)](https://arxiv.org/pdf/1608.06993.pdf) 논문리뷰
+ vision transformer - [AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE(2021)](https://arxiv.org/pdf/2010.11929.pdf) 논문리뷰
+ VIT 코드 리뷰 [code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/Image%20Classification/image_classification_with_vision_transformer.ipynb)
#### 과제1. Classification Project - Kaggle에 있는 classification task를 진행하여 인사이트 뽑아보기.
[과제1을 수행한 개인 블로그 포스팅](https://inhovation97.tistory.com/43)   
[DenseNet논문을 리뷰한 개인 블로그 포스팅](https://inhovation97.tistory.com/47)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week2 - Image Segmentation
+ Segmentation U-net 개념 & 모델 구현 [code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/Image%20Segmentation/Unet_tutorial_origin.ipynb)
+ Dense Block with U-net 실습 [code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/Image%20Segmentation/DensUnet_tutorial_ImgGenerator_Attention_210715.ipynb)
+ SENet - [Squeeze and Excitation Networks(attention)(2018)](https://arxiv.org/pdf/1709.01507) 논문 리뷰
+ PSPNet - [Pyramid Scene Parsing Network(2016)](https://arxiv.org/pdf/1612.01105.pdf) 논문 리뷰
#### 과제2. Segmentation Project - attention block & loss function을 이용하여 성능을 향상하기.
[code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/Image%20Segmentation/ResUnet_tutorial_%EB%8B%A4%EC%A4%91%EB%B6%84%EB%A5%98softmax_with_attention.ipynb)
> 기본적인 U-Net과 attention 모듈을 추가한 residual U-Net의 성능 차이를 비교.   
>    
> ![image](https://user-images.githubusercontent.com/59557720/161203850-05dfbd40-8d0e-4c1c-8244-056b21a57242.png)   
> 두 모델 모두 epoch 20 수준에서의 성능 결과이며, 비교적 segmentation하기 쉬운 이미지로 비교함   
> -> 육안적으로도 위와 같은 큰 차이가 났음.   
>    
> 오른쪽 모델은 loss function을 dice_loss를 이용했으며, binary가 아닌 categorical CE를 이용함   
> -> 분류하기 힘든 경계면 분류가 완화되어 더 나은 성능을 기대   


[Squeeze and Excitation Networks(attention)논문을 리뷰한 개인 블로그](https://inhovation97.tistory.com/48)   
[CBAM(attention)논문을 리뷰한 개인 블로그](https://inhovation97.tistory.com/63)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week 3 - GAN 1
+ DeepLab v3+ - [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation(2018)](https://arxiv.org/pdf/1802.02611.pdf) 논문 리뷰
+ Auto Encoder & Variational Auto Encoder 개념 & 실습   
+ Denoising VAE 실습 [code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/GAN/vae%EA%B3%BC%EC%A0%9C_denoising_autoencoder_%E1%84%89%E1%85%B5%E1%86%AF%E1%84%89%E1%85%B3%E1%86%B8.ipynb)   
+ DCGAN 구현 세부 사항 & 목적 함수
> GAN은 _미술관에 GAN 딥러닝 실전 프로젝트_ 서적을 끝내는 것을 목표로 강의를 진행   
>   
> ![image](https://user-images.githubusercontent.com/59557720/161194974-41882c69-eed1-4f5a-b5c9-34c0f6e6ad2c.png)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week 4 - GAN 2
+ pix2pix - [Image-to-Image Translation with Conditional Adversarial Networks(2017)](https://arxiv.org/pdf/1611.07004.pdf) 논문 리뷰
+ WGAN ( Wasserstein loss, Lipshitz 제약 ) 개념   
+ Cycle GAN ( 여러개의 loss를 이용 ) 개념 & 논문 리뷰 & 실습   
+ Neural Style Transfer - [Image Style Transfer Using Convolutional Neural Networks(2015)](https://arxiv.org/pdf/1508.06576.pdf) 논문 리뷰 & 실습 [code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/GAN/Style_transfer_20210802.ipynb)   
+ SR GAN - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network(2017)](https://arxiv.org/pdf/1609.04802.pdf) 논문 리뷰 & 실습[code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/GAN/SRGAN_20210615.ipynb)   
#### 개인 프로젝트 1 - fine tuning으로 style transfer의 style을 더욱 강력하게 입혀보기
> vggnet을 fine tuning하여 심슨 이미지를 분류하는 모델을 만든 뒤, 해당 모델의 ConV Network을 이용하여 좀 더 강한 심슨풍의 이미지를 유도해봤습니다!   
> [VggNet fine tuning 적용 코드](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/GAN/project1/pretraining_style_transfer.ipynb)   
> [style transfer 적용 코드](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/GAN/project1/style_transfer_in_pytorch.ipynb)   
>   
> ![image](https://user-images.githubusercontent.com/59557720/161210449-88875252-8fbd-446c-ab45-3e27a79e5024.png)
>   
> 보다시피 fine tuning 하면 확실히 style이 강하게 입혀집니다.   
> 우연치 않게 content image와 style image 모두 벽에 액자가 걸려있었는데, 같은 객체로 인식되어 만화 그대로 액자가 입혀져 아주 만족스러웠습니다.   
>    
> 여러개의 이미지를 추론했는데, 머리 있는 사람은 캐릭터로 인식하지 못한 반면 머리가 없는 주호민은 캐릭터로 인식하여 노란색이 입혀짐!   
>    
> 오히려 finetuning 과정에서 심슨 풍의 이미지를 과적합시키고 싶었지만, 큰 데이터셋이나 개발 환경 등이 너무 아쉬웠음.
> 


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week 5 - Object Detection
+ R-CNN -> SPP-Net -> Fast R-CNN -> Faster R-CNN(RPN)
+ YOLO -> SSD -> Retina Net 
+ 위 순서대로  One-stage detector와 Two-stage detector를 발전 계보 순서대로 공부함.
+ FPN - [Feature Pyramid Networks for Object Detection(2017)](https://arxiv.org/pdf/1612.03144)
+ YOLO v5 실습 [code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/Object_detection/train_yolov5_pistols_dataset.ipynb)   

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week 6 - 추가 요청 논문 리뷰 & 개인 프로젝트 2
+ Real-ESRGAN - [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data(2021)](https://arxiv.org/pdf/2107.10833) 논문리뷰   
+ Focal Loss - [Focal Loss for Dense Object Detection(2017)](https://arxiv.org/pdf/1708.02002) 논문리뷰   
+ Tesla AI day - 마침 8월말 테슬라가 AI day에서 자사의 자율 주행 기술을 학회 발표처럼 상세히 oral presentation을 하여 강사님과 함께 들으며 기술에 대한 이야기를 나눔 - vision으로 인코딩한 visual 정보를 LSTM으로 시퀀셜하게 다가가 시뮬레이션 map을 만들어 신기했음.   

#### 개인 프로젝트 2 - YOLOv5 모델을 활용하여 사람 나이 예측 모델을 구현해보기.
> [Roboflow](https://roboflow.com/)플랫폼을 활용하여 [CelebA (사람 얼굴 데이터셋)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 약 800여 장을 직접 라벨링 한 뒤, 모델을 학습하여 나이를 예측하는 pretrained FC Layer를 통해 2stage로 실시간 예측 모델을 계획    
>    
> ![image](https://user-images.githubusercontent.com/59557720/161425381-05068a3e-7c7b-43be-b787-07140adb96d3.png)
>    
> 하지만 fc layer를 위한 age 데이터셋을 확보하는데에 실패... 아쉬운대로 이미 배포된 cv2라이브러리 내에 존재하는 아주 오래된 caffe로 설계한 모델을 활용하여, 이미지 결과물을 도출함.   
>    
> ~~프로젝트 계획에 있어서도 데이터셋 이슈를 염두로 짜야한다는 교훈...~~



