# Get-an-education-for-Computer-Vision
📜 Complete In-depth course in Computer Vision

[서울ICT이노베이션스퀘어](https://ict.eksa.or.kr/portal/applyconfirm_ict/main.user?paramMap.finalGbn=N)에서 교육생을 선발하여 진행하는 인공지능 심화 시각지능 프로젝트 개발 과정을 수료함.
> 2021.07.05 ~ 2021.08.30 기간동안 총 160시간의 교육과정 수료   
> Computer Vision의 흐름과 최신 기술 및 동향을 공부   
> __개인적으로 크게 성장할 수 있는 기회였으며, CV분야에 흠뻑 빠져 너무나도 즐겁게 공부했던 기억이 납니다.__   
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## week1 - Image Classification
+ 기본적인 NN의 순전파/역전파
+ transfer learning & Data Augmentation
+ DenseNet - [Densely Connected Convolutional Networks(2017)](https://arxiv.org/pdf/1608.06993.pdf) 논문리뷰
+ vision transformer - [AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE(2021)](https://arxiv.org/pdf/2010.11929.pdf) 논문리뷰
#### 과제1. Classification Project - Kaggle에 있는 classification task를 진행하여 인사이트 뽑아보기.
[과제1을 수행한 개인 블로그 포스팅](https://inhovation97.tistory.com/43)   
[DenseNet논문을 리뷰한 개인 블로그 포스팅](https://inhovation97.tistory.com/47)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week2 - Image Segmentation
+ Segmentation U-net 개념 & 모델 구현 [code]()
+ Dense Block 실습 [code]()
+ SENet - [Squeeze and Excitation Networks(attention)(2018)](https://arxiv.org/pdf/1709.01507) 논문 리뷰
+ PSPNet - [Pyramid Scene Parsing Network(2016)](https://arxiv.org/pdf/1612.01105.pdf) 논문 리뷰
#### 과제2. Segmentation Project - attention block & loss function을 이용하여 성능을 향상하기.
[code]()
> attention block의 효과는 미미했으며, Loss Function으로는 mean IOU와 Dice coefficient를 이용해봤는데 다른 metric이라 비교하기는 애매하지만 dice_loss가 육안으로 더 나아보이고, f1 score(이진 분류에 한함)와 동일한 수식이므로 좀 더 합리적인 것 같다.

[Squeeze and Excitation Networks(attention)논문을 리뷰한 개인 블로그](https://inhovation97.tistory.com/48)   
[CBAM(attention)논문을 리뷰한 개인 블로그](https://inhovation97.tistory.com/63)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week 3 - GAN 1
+ DeepLab v3+ - [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation(2018)](https://arxiv.org/pdf/1802.02611.pdf) 논문 리뷰
+ Auto Encoder & Variational Auto Encoder 개념 & 실습 [code]()
+ DCGAN 구현 세부 사항 & 목적 함수
> GAN은 _미술관에 GAN 딥러닝 실전 프로젝트_ 서적을 끝내는 것을 목표로 강의를 진행   

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week 4 - GAN 2
+ pix2pix - [Image-to-Image Translation with Conditional Adversarial Networks(2017)](https://arxiv.org/pdf/1611.07004.pdf) 논문 리뷰
+ WGAN ( Wasserstein loss, Lipshitz 제약 ) 개념
+ Cycle GAN ( 여러개의 loss를 이용 ) 개념 & 논문 리뷰 & 실습 [code]()
+ Neural Style Transfer - [Image Style Transfer Using Convolutional Neural Networks(2015)](https://arxiv.org/pdf/1508.06576.pdf) 논문 리뷰
+ SR GAN - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network(2017)](https://arxiv.org/pdf/1609.04802.pdf) 논문 리뷰   
#### 개인 프로젝트 1 - fine tuning으로 style transfer의 style을 더욱 강력하게 입혀보기
> vggnet을 fine tuning하여 심슨 이미지를 분류하는 모델을 만든 뒤, 해당 모델의 ConV Network을 이용하여 좀 더 강한 심슨풍의 이미지를 유도해봤습니다!
img

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week 5 - Object Detection
+ R-CNN -> SPP-Net -> Fast R-CNN -> Faster R-CNN(RPN)
+ YOLO -> SSD -> Retina Net 
+ 위 순서대로  One-stage detector와 Two-stage detector를 발전 계보 순서대로 공부함.
+ FPN - [Feature Pyramid Networks for Object Detection(2017)](https://arxiv.org/pdf/1612.03144)
+ YOLO v5 실습 [code]()

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## week 6 - 추가 요청 논문 리뷰 & 개인 프로젝트 2
+ Real-ESRGAN - [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data(2021)](https://arxiv.org/pdf/2107.10833) 논문리뷰   
+ Focal Loss - [Focal Loss for Dense Object Detection(2017)](https://arxiv.org/pdf/1708.02002) 논문리뷰   
+ Tesla AI day - 마침 8월말 테슬라가 AI day에서 자사의 자율 주행 기술을 학회 발표처럼 상세히 oral presentation을 하여 강사님과 함께 들으며 기술에 대한 이야기를 나눔 - vision으로 인코딩한 visual 정보를 LSTM으로 시퀀셜하게 다가가 시뮬레이션 map을 만들어 신기했음.   

#### 개인 프로젝트 1 - obj detection 모델을 활용하여 사람 나이 예측 모델을 구현해보기.
> [Roboflow](https://roboflow.com/)플랫폼을 활용하여 [CelebA (사람 얼굴 데이터셋)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 약 800여 장을 직접 라벨링 한 뒤, 모델을 학습하여 나이를 예측하는 pretrained FC Layer를 통해 2stage로 실시간 예측 모델을 계획    
>    
> 하지만 fc layer를 위한 age 데이터셋을 확보하는데에 실패... 아쉬운대로 이미 배포된 모델들을 활용하여, 결과물을 도출함.   
>    
> ~~프로젝트 계획에 있어서도 데이터셋 이슈를 염두로 짜야한다는 교훈...~~
> img



