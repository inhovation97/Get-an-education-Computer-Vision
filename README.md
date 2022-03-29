# Get-an-education-Computer-Vision
📜 Complete In-depth course in Computer Vision

[서울ICT이노베이션스퀘어](https://ict.eksa.or.kr/portal/applyconfirm_ict/main.user?paramMap.finalGbn=N)에서 교육생을 선발하여 진행하는 인공지능 심화 시각지능 프로젝트 개발 과정을 수료함.
> 2021.07.05 ~ 2021.08.30 기간동안 총 160시간의 교육과정 수료
> Computer Vision의 흐름과 최신 기술 및 동향을 공부
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## week1 - Image Classification
+ 기본적인 NN의 순전파/역전파
+ transfer learning & Data Augmentation
+ DenseNet 논문리뷰 - [Densely Connected Convolutional Networks(2017)](https://arxiv.org/pdf/1608.06993.pdf)
+ vision transformer - [AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf)
#### 과제1. Classification Project - Kaggle에 있는 classification task를 진행하여 인사이트 뽑아보기.
[과제1을 수행한 개인 블로그 포스팅](https://inhovation97.tistory.com/43)   
[DenseNet논문을 리뷰한 개인 블로그 포스팅](https://inhovation97.tistory.com/47)




## week2 - Image Segmentation
+ Segmentation U-net 개념 & 모델 구현 [code]()
+ Dense Block 실습 [code]()
+ Squeeze and Excitation Networks(attention) 개념
+ PSPNet - [Pyramid Scene Parsing Network](https://arxiv.org/pdf/1612.01105.pdf) 논문 리뷰
#### 과제2. Segmentation Project - attention block & loss function을 이용하여 성능을 향상하기.
[code]()
-> attention block의 효과는 미미했으며, Loss Function으로는 mean IOU와 Dice coefficient를 이용해봤는데 다른 metric이라 비교하기는 애매하지만 dice_loss가 육안으로 더 나아보이고, f1 score(이진 분류에 한함)와 동일한 수식이므로 좀 더 합리적인 것 같다.

[Squeeze and Excitation Networks(attention)논문을 리뷰한 개인 블로그](https://inhovation97.tistory.com/48)
[CBAM(attention)논문을 리뷰한 개인 블로그](https://inhovation97.tistory.com/63)




## week 3 - GAN 1
+ DeepLab v3+ - [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf) 논문 리뷰
+ Auto Encoder & Variational Auto Encoder 개념 & 실습 [code]()
+ DCGAN 구현 세부 사항 & 목적 함수
> GAN은 _미술관에 GAN 딥러닝 실전 프로젝트_ 서적을 끝내는 것을 목표로 강의를 진행   




## week 4 - GAN 2
+ WGAN ( Wasserstein loss, Lipshitz 제약 ) 개념
+ Cycle GAN ( 여러개의 loss를 이용 ) 개념 & 실습 [code]()
+ Neural Style Transfer - [Image Style Transfer Using Convolutional Neural Networks](https://arxiv.org/pdf/1508.06576.pdf) 논문 리뷰
+ SRGAN (Super Resolution GAN)
#### 개인 프로젝트 - fine tuning으로 style transfer의 style을 더욱 강력하게 입혀보기
> vggnet을 fine tuning하여 심슨 이미지를 분류하는 모델을 만든 뒤, 해당 모델의 ConV Network을 이용하여 좀 더 강한 심슨풍의 이미지를 유도해봤습니다!
img




## week 5 - obj detection
+ object detection 





