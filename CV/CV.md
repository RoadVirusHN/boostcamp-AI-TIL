[TOC]

# Computer Vision(CV)

> 네이버 AI 부스트 캠프의 CV 강의를 정리한 내용입니다.

## Image Classification 1

### Course overview

Artificial Intelligence(AI) : 사람의 지능을 컴퓨터 시스템으로 구현하는 것

사람의 학습은 오감을 조합해서 사용하는 multi-modal perception과 비슷하다.

뿐만 아니라 사회적 감각(표정 살피기, 관계 맺기, 의도 파악하기) 등의 복합적 감각이 존재

이중 시각적 능력이 기초적이고 중요한 능력이다.

![image-20210308093803218](CV.assets/image-20210308093803218.png)

**[img. 복잡한 인간의 인지]**

|          인간의 시각적 처리 VS 컴퓨터의 시각적 처리          |
| :----------------------------------------------------------: |
| ![image-20210308093956611](CV.assets/image-20210308093956611.png) |
| ![image-20210308093950605](CV.assets/image-20210308093950605.png) |

**[fig. 사진의 representation(자료구조, 여기서는 처리 결과) 구하기]**

또한, 우리의 시각적 능력 또한 이런 머신러닝 구조의 CV와 비슷하다.

![image-20210308094608496](CV.assets/image-20210308094608496.png)

**[img. 대처 환각, 거꾸로 된 사람은 많이 보지 못하기 때문(= 학습 편향)에 어색함을 느끼지 못한다.]**

|              Machine Learning vs Deep Learning               |
| :----------------------------------------------------------: |
| ![image-20210308094749066](CV.assets/image-20210308094749066.png) |
| ![image-20210308094810716](CV.assets/image-20210308094810716.png) |

**[img. 특징 추출까지 처리가 가능해진 Deep Learning 분야]**

특징 추출을 사람이 설계하지않고 Gradient-Descend를 통해 End-to-End로 학습함으로, 편향과 오류를 줄여 최근 CV기술이 고도화 됨

이번 강의에서

- 기본적인 CV Task
- CV의 딥러닝 기술
- 시각 정보 + 다른 감각 데이터의 융합
- Conditional generative model
- Visualization Tool

등을 배울 것이다.

### Image classification(영상 분류)

Classifier (분류기) 업무는 입력된 영상의 카테고리, 클래스를 분류하는 mapping 업무이다.

![image-20210308095502653](CV.assets/image-20210308095502653.png)

**[img. Classifier 예시]**

만약, 대량의 이미지 데이터가 있다면, 비슷한 사진끼리 모아 구별하는 *k-Nearest Neighbors(k-NN)*를 통하여 구별할 수 있다.

- query data point의 주변 reference point를 이용해 구분하는 방법

![image-20210308100000964](CV.assets/image-20210308100000964.png)

**[img. K-NN 예시 이미지]**

하지만, 데이터 수가 너무 많으면 많은 만큼 시간과 메모리가 부족하게 된다.

이를 방지하기 위해 Neural Networks를 이용해 Compress 된 모델을 통해 구별하게 된다.

![image-20210308100345559](CV.assets/image-20210308100345559.png)

**[img. 1층의 fully-connected layer을 이용한 구분]**

영상의 모든 픽셀을 Input으로 넣어 구분할 수도 있지만, 이 역시 성능과 메모리 소모가 크다.

![image-20210308101409313](CV.assets/image-20210308101409313.png)

**[img. FCN과 CNN의 비교]**

CNN은 전체 pixel이 아닌 이미지의 일부를 지역적으로 Window Sliding 방식으로 하나의 feature로 추출하고(Local feature learning), 파라미터 공유를 통하여 

1. 성능과 메모리의 요구를 줄이고 
2. 오버피팅을 줄일고
3. 사진 일부만으로도 사진을 구별할 수 있게 됬다.

이러한 장점 덕분에 많은 Computer Vision 업무에 기본이 됨.

![image-20210308102242021](CV.assets/image-20210308102242021.png)

**[img. CNN의 여러 사용]**

### CNN architectures for image classification 1

![image-20210308104319839](CV.assets/image-20210308104319839.png)

**[img. AlexNet과 VGGNet의 등장]**

- 딥러닝 CV의 등장을 알린 두 모델

1. AlexNet

![image-20210308104438510](CV.assets/image-20210308104438510.png)

**[img. AlexNet의 구조]**

- 최초의 심플한 CNN 구조인 LeNet-5의 구조에서 더 깊은 층, 더 많은 데이터셋, ReLU 활성 함수와 drop out과 같은 regularization 기술을 이용한 모델
- 당시 GPU 메모리의 한계로 인해 2갈래로 나누어 학습한 뒤 중간에 Activation map으로 Cross Communication 해줌

```python
nn.conv2d(3, 96, kernel_size=11, stride=4, padding=2) # 11x11 Conv(96), stride 4 Layer
# 이미지 크기가 커지면서, Receptive field를 크게 만들어주기 위해 11x11로 시작, 최근에는 더이상 사용 안함
nn.ReLU(inplace=True)

nn.MaxPool2d(kernel_size=3, stride=2)# 3x3 MaxPool, stride 2 Layer

nn.conv2d(96, 256, kernel_size=5, padding=2) # 5x5 Conv(256), stride 2 Layer
nn.ReLU(inplace=True)

nn.MaxPool2d(kernel_size=3, stride=2)

nn.conv2d(256, 384, kernel_size=3, padding=1)# 3x3 Conv(384), pad 1 Layer
nn.ReLU(inplace=True)

nn.conv2d(384, 384, kernel_size=3, padding=1)
nn.ReLU(inplace=True)
nn.conv2d(384, 256, kernel_size=3, padding=1)
nn.ReLU(inplace=True)

nn.MaxPool2d(kernel_size=3, stride=2)

torch.flatten(x, 1) # Fully connected Layer에 넣기 전에 다차원의 Tensor를 1차원 Tensor으로 길이를 길게 늘어 뜨리기(벡터화)
# AlexNet에서 사용한 방법

# nn.AdaptiveAvgPool2d((6,6))
# 비슷한 방법이지만, 길게 늘어뜨리지 않고 평균을 내어 같은 길이의 1차원으로 바꿈

nn.Dropout() # Dense(4096)
nn.Linear(256*6*6, 4096) # 2stream 구조가 아니라 1개로 통합했으므로 2048 *2 = 4096
nn.ReLU(inplace=True)

nn.Dropout()
nn.Linear(4096, 4096)
nn.ReLU(inplace=True)

nn.Linear(4096, 1000)
```

**[code. AlexNet의 코드구현]**

- 시간이 흘러 GPU 메모리가 늘어나 2 stream 구조로 구현하지 않음
- LRN(Local Response Normalization) 구현 안함
  - 더이상 사용하지 않는 방법, Batch normalization으로 대체됨
  - Activation map 이후, 명암을 normalization 해주는 역할

Receptive field란?

layer output 값을 만들기위해 Input image에서 CNN layer가 참조한 공간, 클 수록 이미지의 많은 부분을 참조한 것이다. 

여러 층이 중첩되도 처음 image에서 확인한 부분이 Receptive field이다. 

위 구조에서는 전체를 11x11 conv로 Input 이미지 전부를 Receptive field로 삼았다.

![image-20210308111832639](CV.assets/image-20210308111832639.png)

**[img. Receptive Field 도식화]**

KxK conv stride 1 layer와 PxP pooling layer를 통과한 경우의 Receptive field의 크기는

(P+K-1)x(P+K-1)이다.

2. VGGNet

![image-20210308112229605](CV.assets/image-20210308112229605.png)

**[img. VGGNet의 구조]**

- AlexNet보다 깊고(16, 19Layer)
- 더욱 심플한 구조이며
  - Loca Response Normalization(LRN) 사용 안함
  - conv filter layer와 max pooling layer의 크기를 각각 3x3, 2x2만 한하여 사용(가장 큰 특징)
    - 이를 stack하여 큰 Receptive Size를 얻으면서 더 깊고 복잡하면서 파라미터 수는 줄여 성능과 정확도를 동시에 잡을 수 있다.
- 더욱 좋은 성능과 일반화(Generlization)을 내는 모델

이외에는 AlexNet과 비슷하다.

- ReLU 사용,  Input에서 224x224 RGB 이미지를 Normalization(RGB 평균값을 RGB 값에서 빼줌)하여 넣어줌

## Annotation data efficient learning

- 질좋은 데이터셋은 성능에 큰 영향을 미치지만 확보하거나 만드는데 큰 어려움이 따른다.
- CV에서의 데이터 부족 완화 방법을 알아보자. 

### Data augmentation

손쉽게 데이터셋을 늘릴 수 있는 방법

|      |                          여러 물체                           |                             장소                             |                             공원                             |
| ---- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 그림 | ![image-20210308133559123](CV.assets/image-20210308133559123.png) | ![image-20210308133608690](CV.assets/image-20210308133608690.png) | ![image-20210308133615833](CV.assets/image-20210308133615833.png) |
| 편향 |          겹쳐 보면 정면에 특정 각도로 물체들이 위치          |  장소 사진들을 겹쳐보면 해안선, 물체, 건물 등의 위치가 겹침  |          사람을 위주로 찍게 되어 중앙에 사람이 위치          |

**[fig. 편향의 예시]**

Dataset들은 인간의 필요에 의해 편향된 채로 촬영되게 되며, 이는 현실의 데이터와 괴리를 준다.

| Samples in the training set                                  | Real data distribution                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20210308133924729](CV.assets/image-20210308133924729.png) | ![image-20210308133920291](CV.assets/image-20210308133920291.png) |

애매한 데이터들이 여러 class와 겹치는 방식인 현실 데이터와 달리 sample data들은 확실하고 명확한 경우가 많으므로 이로 인해 모델에 혼란이 올 수 있다.

예를 들면, 많은 사람들이 밝은 조명 아래에서 사진을 찍는다, 그리고 달은 어두운 밤하늘에만 찍힌다. 만약 이를 데이터셋으로 학습한 모델이 어두운 곳에 찍힌 사람을 보면, 달로 착각할 수도 있다.

![image-20210308134801650](CV.assets/image-20210308134801650.png)

**[img. Augmenation의 예시]**

이를 막기위해, 밝기 바꾸기, 회전, crop, 일부 가리기, 채도 변경 등의 방법으로 이미지를 바꾸어 현실 데이터와 비슷하게 만들면서 데이터셋 크기를 늘릴 수 있다.

OpenCV, NumPy 등에서 library로 활용할 수 있다.

|                                                              |                                                              |                                                              |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image-20210308140744851](CV.assets/image-20210308140744851.png) | ![image-20210308140858344](CV.assets/image-20210308140858344.png) | ![image-20210308140807904](CV.assets/image-20210308140807904.png) |
|                         Rotate, Flip                         |                    Brightness adjustment                     |                             Crop                             |
| ![image-20210308140751671](CV.assets/image-20210308140751671.png) | ![image-20210308140824819](CV.assets/image-20210308140824819.png) | ![image-20210308140830735](CV.assets/image-20210308140830735.png) |
|                         Rotate, Flip                         |                    Affine transformation                     |                            CutMix                            |

**[table. 여러 종류의 augmentation 종류]**

```python
def brightness_augmentation(img):
    # numpy array img has RGB value(0~255) for each pixel
    img[:,:,0] = img[:,:,0]+100 # add 100 to R value
    img[:,:,1] = img[:,:,1]+100 # add 100 to G value
    img[:,:,2] = img[:,:,2]+100 # add 100 to B value
    
    img:[:,:,0][img[:,:,0]>255] = 255 # clip R values over 255
    img:[:,:,1][img[:,:,1]>255] = 255 # clip G values over 255
    img:[:,:,2][img[:,:,2]>255] = 255 # clip B values over 255
    return img
```

**[code. 밝기 조절 Augmentation 코드]**

```python
img_rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
img_flipped = cv2.rotate(image, cv2.ROTATE_180)
```

**[code. 회전, 뒤집기 Augmentation 코드]**

```python
y_start = 500 # y pixel to start cropping
crop_y_size = 400 # cropped image's height
x_start = 300 # x_pixel to start cropping
crop_x_size = 800 # cropped image's width
img_cropped = image[y_start:y_start+crop_y_size, x_start : x_start + crop_x_size, :]
```

**[code. Crop Augmentation 코드]**

```python
rows, cols, ch = image.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1, pts2)
shear_img = cv2.warpAffine(image, M, (cols,rows))
```

**[code. Affine Transformation Augmentation]**

Affine Transformation : 사각형 이미지를, 기울어진 평행 사변형(parallelogram)의 형태로 바꿈 + rotation 함

- shear transformation 이라고도 함

RandAugment

- Augmenation의 종류, 강도(얼마나 밝은가, 얼마나 기울어졌는가 등)에 따라 모델 성능이 달라지므로, 이를 parameter로 사용할 수 있다.
- 랜덤으로 Augmentation을 적용한 뒤, 평가하는 과정을 거침

![image-20210308144014875](CV.assets/image-20210308144014875.png)

**[img. ShearX & AutoContrast 9 Randaug 예시]**

- 적용할 Augmentation, 적용 강도, 2가지가 Parameter로 적용

![image-20210308144310891](CV.assets/image-20210308144310891.png)

**[img. RandAugmentation 사용에 의한 성능 향상]**


### Leveraging pre-trained information
pre-trained 된 모델을 활용하는 방법

#### Transfer learning

Transfer learning: 한 데이터셋에서 배운 지식을 다른 데이터셋, 다른 Task에서 활용하는 기술

- 이를 통하여 작은 데이터셋으로도 높은 정확도를 자랑한다.

1. Transfer knowledge from a pre-trained task to a new task(같은 Layer, 다른 Task에 활용)

![image-20210308153411813](CV.assets/image-20210308153411813.png)

**[img. Transfer model(좌)를 원하는 업무에 맞게 FCL을 변경]**

- 마지막 Fully connected Layer만 바꾼 뒤, 이전 Convolution Layer의 Weight는 그대로 둔채 바꾼 층만 학습 시켜 활용하는 방법
- pre-trained 모델의 feature가 그대로 유지됨

2. Fine-tuning the whole model

![image-20210308153626775](CV.assets/image-20210308153626775.png)

**[img. FCL을 변경 후 Convolution Layer도 Low lr로 학습]**

- 마지막 층을 바꾼 뒤, 기존 층은 낮은 Learnig rate, 새로운 층은 High learning rate를 유지하며 학습
- 자신의 Task에 맞게 기존 모델을 조금 수정가능

#### Knowledge distillation

pre-trained 모델의 예측값을 활용해 다른 모델을 학습시키는 방법

Teacher-student learning

보통 두가지 목적으로 쓰임

1. 더 경량화된 모델을 만들어, 기존 모델 보다 경량화에 사용
2. unlabeld dataset의 pseudo-labelling에 사용(레이블링 안된 데이터셋에 라벨링)



학습 구조에 따라 2가지로 나뉜다.

1. Teacher-student network structure

- student 모델(보통, Teacher Model보다 경량화되어 있다.)이 Teacher 모델의 output을 따라하게끔 학습시킴
- Unsupervised learning으로, label 되지 않은 dataset을 사용한다.
- 두 결과를 비교하여 KL divergence Loss를 통해 Loss 값을 구한뒤, student model로만 backpropagation이 진행되게 된다.
  - KL div Loss: 2개의 분포의 거리(=얼마나 비슷한가)를 측정

![image-20210308155338677](CV.assets/image-20210308155338677.png)

**[img. Teacher-student network structure]**



2. Knowledge distillation

![image-20210308155357681](CV.assets/image-20210308155357681.png)

**[img. Knowledge distillation structure]**

- Labeled된 데이터셋을 사용할 때 사용하는 구조
- Student Model과 Teacher Model의 차이를 Distillation Loss, Student Model과 Ground Truth(Label)과의 차이를 Student  Loss라 정의 한다.
- 이때 Student 모델은 Soft label을 이용한 Soft Prediction을 이용한다.
  - Hard label(One-hot vector) : dataset의 label값처럼, 하나의 명확한 정답을 가짐
  - Soft label: model의 softmax를 통과한 뒤 기본 출력값, 여러 정답에 float 값을 가지는 가중치같은 형태
  - 이를 이용해 단순 정답이 아닌, 어느 정도로 정답에 근접했는가 등의 추가적인 정보를 사용하여 학습할 수 있다.


$$
Hard\ label:\begin{pmatrix}Bear\\Cat\\Dog\end{pmatrix}=\begin{pmatrix}0\\1\\0\end{pmatrix}\\
Soft\ label:\begin{pmatrix}Bear\\Cat\\Dog\end{pmatrix}=\begin{pmatrix}0.14\\0.8\\0.06\end{pmatrix}
$$
**[math. Hard label과 Soft label의 예시]**

- 또한 Knowledge distilation에서의 Distillation Loss를 구할 때, Softmax 함수에 temperature(T)를 이용하여 기존의 output보다 Smoothing한 결과를 사용할 수 있다.
  - 이를 이용해 좀더 결과값에 많은 정보를 포함할 수 있다.
  - 예를 들어 기존의 softmax(5,10) = (0.0067, 0.9933)이라면
  - t=100인 softmax(5,10) = ( 0.4875, 0.5125)로 비슷한 값을 smoothing 된다.
  - 전자의 결과의 0,0067은 무시될 정도로 작은 정보지만 후자의 경우는 무시못할 정보가 된다.


$$
Normal\ Softmax(=Hard\ Prediction):\frac{\exp(z_i)}{\sum_j\exp(z_j)}\\
Softmax\ with\ temperature\ T(=Soft\ Prediction):\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$
**[math. softmax with temperature]**

- Distillation Loss를 구할 때, Teacher Model의 soft Label의 세부정보(Semantic information)은 고려하지 않는다.
  - 교육받고 바꿔야할 모델은 Student Model 이기 때문에

Distillation Loss의 경우, Teacher model vs Student model의 차이를 의미

- KL divergence loss 사용: 비교 하는 두 값이 0~1사이의 값들의 분포이므로

Student Loss의 경우, 실제 답과 student model의 정답을 비교

- CrossEntropy 사용: True label의 경우 one-hot vector 형태이므로



마지막으로 위에서 구한 두 loss들의 가중치 합을 통해 loss를 구한 뒤, student model만, Backpropation을 한다.

### Leveraging unlabled dataset for training

#### Semi-supervised learning

- labeled된 적은 수의 데이터와, label되지 않은 많은 양의 데이터를 활용하는 방법
  - 즉 unsupervised + Fully supervised = semi-supervised

![image-20210308224613244](CV.assets/image-20210308224613244.png)

1) 레이블링된 데이터셋으로 모델 형성

2) 형성된 모델로 레이블링되지 않은 데이터셋 레이블링

3) 그렇게 레이블링된 데이터셋과 기존의 라벨링된 데이터셋으로 새로운 모델 형성

#### Self-training

앞서 배웠던 Data Augmentation, Knowledge distillation, Semi-supervised learning을 이용한 방법

![image-20210308224943394](CV.assets/image-20210308224943394.png)

**[img. self-training의 단계]**

1. lable된 데이터셋으로 Teacher model을 형성한다.
2. 해당 Teacher model로 unlabled된 model을 Pseudo-labeled data로 만든다.
3. lable된 데이터셋 + pseudo-lable된 데이터셋을 augmentation 한 것을 통하여 새로운 Teacher model을 형성한다
   - 이때 주로 사용하는 augmentation  방법이 RandAugment
4. 새로 형성된 Teacher 모델로 2번부터 4번까지 반복한다. 

![image-20210308225809995](CV.assets/image-20210308225809995.png)

**[img. 압도적인 성능을 자랑하는 self-training 모델(빨간색)]**

## Image classification 2

### Problems with deeper layers

성능 향상을 위해 딥러닝 layer의 층을 높게 쌓으면서 다음과 같은 문제가 생겼다

1. Gradient vanishing/exploding 문제
2. Computationaly complex
3. 한때 overfitting 문제로 착각했던 Degradation problem

![image-20210309100228831](CV.assets/image-20210309100228831.png)

**[img. Vanishing gradient 문제의 도식]**

### CNN architectures for image classification 2

#### GoogLeNet

2015년에 발표된 Inception 모듈을 활용한 CV 모델

Inception module이란?

- 이전 층에서의 결과값에 여러개의 필터를 적용한 뒤, Concatenate하는 layer

  - 1x1, 3x3, 5x5 Convolution filter, 3x3 max pooling layer를 적용

  

![image-20210309112126930](CV.assets/image-20210309112126930.png)

**[img. Inception module의 예시]**

이때, 여러 필터의 적용에 의해 parameter수가 증가하자, 1x1 convolution layer(Bottleneck layer)을 추가하여 파라미터 수를 줄이는 시도를 함

- 우측의 Dimension Reduced version에 추가된 1x1 convolution layer를 의미

![image-20210309112308655](CV.assets/image-20210309112308655.png)

**[img. 1x1 convolution layer의 연산 결과]**

GoogLeNet의 전체적인 구조를 살펴보면 다음과 같다.

1. Stem network: 기본적인 convolution network

![image-20210309115934238](CV.assets/image-20210309115934238.png)

**[img. Stem network 부분]**

2. Stacked inception modules: 위에 설명한 Inception 모듈을 쌓아놓은 부분

![image-20210309120018533](CV.assets/image-20210309120018533.png)

**[img. Stacked inception modules 부분]**

3. Auxiliary classifiers 

- Vanishing gradient 문제를 해결하기 위한 부분
- 중간의 결과값을 한번 예측값으로 삼고, loss 값을 계산하여 중간부터 backpropagation을 진행한다
- training에서만 사용하고 testing 단계에서는 사용하지 않는다.

![image-20210309120037565](CV.assets/image-20210309120037565.png)

**[img. Auxiliary classifiers 부분]**

![image-20210309120621091](CV.assets/image-20210309120621091.png)

**[img. 더욱 자세한 Auxiliary classifier]**

#### ResNet

현재까지 기본 backbone으로 쓰이곤 하는 좋은 모델

- 최초로 인간 보다 나은 성능을 달성(에러율 기준)
- 기존의 모델보다 압도적으로 깊은 층의 갯수(152 Layer)

기존의 연구에서는 층이 깊을 수록 오히려 성능이 떨어지는 문제를 Overfitting 문제라고 오판하였다.

![image-20210309122016623](CV.assets/image-20210309122016623.png)

**[img. 층의 갯수에 따른 에러율, 높을 수록 안좋음]**

Overfitting의 문제였다면, training error는 점점 나아져야하고, test error가 나빠져야 하지만, 둘다 성능이 나빠졌기 때문이다.

따라서 Resnet에서는 이를 Overfitting이 아닌 Optimization(최적화)의 문제라고 보았다.

ResNet의 연구 가설은 다음과 같다.

![image-20210309123313947](CV.assets/image-20210309123313947.png)

**[img. Residual block과 Plain layer의 차이]**

기존의 Plain layer의 경우 층이 깊어질 수록 복잡해진 H(x)에 X를 보존하면서 학습하기 힘들었다.

하지만 Residual block에서는 identity X를 F(X)에 더한 것을 H(X)로 삼으면서, X의 정체성이 뚜렷히 남은 상태에서, 분할정복 통해 최적화된 학습을 할 수 있다.

- 분할정복-> (F(x), X의 weight를 따로 구해서 더하면 되니까?)
- Target function : $H(x)=F(x)+x$
- Residual function : $F(x)=H(x)-x$ 

이를 위해 *Shortcut connection 또는 Skip connection*을 통해 x를 layer을 넘어 더해주어 Backpropagation 시 뛰어넘어 gradeint를 구할 수 있게 하였다.

- 이를 통해 Gradient vanishing 문제를 해결함

|                        Residual 구조                         |                      경로를 풀어본 구조                      |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image-20210309124015987](CV.assets/image-20210309124015987.png) | ![image-20210309124032924](CV.assets/image-20210309124032924.png) |

**[img. 같은 Residual 구조의 경로 풀이]**

이론상, Residual 구조를 통하여 생기는 경로는 층이 깊이 n에 따라 $O(2^n)$개 만큼 증가한다.

이 경로를 통해 backpropagation이 가능하므로 복잡한 학습을 해결 가능하다.

ResNet의 전체적인 구조는 다음과 같다.

1. He initialization conv layer

![image-20210309130058413](CV.assets/image-20210309130058413.png)

**[img. Resnet의 첫 시작 부분]**

- 첫 layer의 output은 앞으로 계속 identity connection을 통하여 더해질 것이므로, 최적화를 위해 단순하고 작은 크기의 output을 내놓아야 한다.
- 따라서 He initialization이라는 간단하고 ResNet을 위해 고안된 initialization을 이용한다.

2. Stacked residual blocks 부분

![image-20210309130304826](CV.assets/image-20210309130304826.png)

**[img. Stack residual blocks]**

- 모두 3x3 conv layer로 이루어져 있으며, Batch normalization이 매 layer 끝에 이루어진다
- 일정 블록 이후(색이 바뀌는 부분), 채널 수는 2배로 늘리고, 채널 해상도는 stride를 2로 잡아 줄이는 구간이 존재함

3. Output FC layer

![image-20210309130815026](CV.assets/image-20210309130815026.png)

**[img. Sing FC layer]**

- 하나의 average pooling과  Fully connected layer을 통하여 classfication을 진행

**ResNet 코드**

![image-20210309214256499](CV.assets/image-20210309214256499.png)

**[img. ResNet code stack 수 정의 부분 ]**

![image-20210309214240250](CV.assets/image-20210309214240250.png)

**[img. ResNet code 첫시작, He initialization 부분 ]**

![image-20210309214221377](CV.assets/image-20210309214221377.png)

**[img. ResNet code, stacked residual 부분 ]**

![image-20210309214208739](CV.assets/image-20210309214208739.png)

**[img. ResNet code, Layer 생성 코드]**

![image-20210309214151827](CV.assets/image-20210309214151827.png)

**[img. ResNet code, 마지막 FC층 부분]**

#### Beyond ResNets

1. DenseNet

![image-20210309135815464](CV.assets/image-20210309135815464.png)

**[img. DenseNet 이미지]**

- ResNet과 달리 layer의 output이 이후의 모든 layer의 결과값에 Channel 축을 중심으로 Concatenate되서 합해진다.
- Cocatenate하므로 기존의 값들이 보존된다
2. SENet
- Activation의 결과가 명확해지도록 ouput의 채널축에 가중치를 주는 Attention across channel 방식
- feature의 중요도와 관계가 명확해짐
- Squeeze: global average pooling을 통하여 채널의 공간정보를 없애고(축의 정보 등) 분포를 구함
- Excitation: FC layer 하나로 채널간의 연관성(Weight=attention score)를 구함
- 중요도가 떨어지면 0에 가깝게 중요한 것은 크게 하여 feature의 강조와 무시를 함
3. EfficientNet
- 기존의 Network 알고리즘을 정리함

| 기본 Network                                                 | Width Scailing                                               | Depth Scailing                                               | Resolution Scaling                                           | Compound Scailing                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20210309140728124](CV.assets/image-20210309140728124.png) | ![image-20210309140741008](CV.assets/image-20210309140741008.png) | ![image-20210309140753841](CV.assets/image-20210309140753841.png) | ![image-20210309140806358](CV.assets/image-20210309140806358.png) | ![image-20210309140821739](CV.assets/image-20210309140821739.png) |
| 기준                                                         | 채널의 수를 늘리는 방식                                      | 층의 수를 늘리는 방식                                        | Input의 해상도를 높게 주는 방식                              | 앞의 방법들을 복합한 방식                                    |
|                                                              | GoogLeNet 등                                                 | DenseNet 등                                                  |                                                              | EfficientNet                                                 |

**[table. 기존의 Network의 분류]**

- 각 Scailing들은 파라미터 수, 학습 epoch, 데이터셋의 수에 따라 성능이 오르지않는 구간이 나오는데, 이를 모두 변수(팩터, 어느 정도 비율로 복합하는가)를 주고 복합하여 성능을 크게 상승시킴

- 사람이 찾은 효율적인 다른 구조들, NAS 알고리즘 구조(Neural Architecture Search, 컴퓨터가 효율적인 구조를 찾는 알고리즘)보다 성능이 압도적으로 좋다.

![image-20210309141007253](CV.assets/image-20210309141007253.png)

- 적은 연산으로도 성능이 크게 올라 EfficientNet이다.

4. Deformable convolution

![image-20210309141016999](CV.assets/image-20210309141016999.png)

- 동물, 사람, 등의 형태가 변할 수 있는 사물에 효율적인 구조
- feature를 나타내는 weight와, 이 weight의 위치를 어떠한 방향으로, 어떻게 변형시킬 지 결정하는 offsets를 학습하는 형식
- 기존의 정사각형 형태의 Receptive field와 달리 물체의 형태에 따라서 Receptive field 모양이 변함

### Summary of image classification

![image-20210309142638948](CV.assets/image-20210309142638948.png)

**[img. 앞서 배운 모델들의 비교, 면적은 모델의 크기]**

- AlexNet은 심플하지만 메모리 사용량이 크고 성능이 좋지 않다
- VGGNet은 성능이 낫지만 메모리와 연산을 많이 잡아 먹는다
- GoogLeNet의 최신 구조는 크기도 적고 성능도 좋지만, 구조가 복잡하다
- ResNet은 특출난 것이 없다
- GoogLeNet이 여러모로 좋지만 구조가 너무 복잡하여 VGGNet, ResNet을 기본 모델로 많이 사용한다.



## Semantic segmentation

### Semantic segmentation

이미지 각 픽셀의 어떠한 category에 속하는지 구분하는 문제(ex) 사람 영역, 자동차 영역)

같은 class의 다른 instance에는 관계가 없으며 이를 위한 instance segmentation가 있다.

![image-20210309155939051](CV.assets/image-20210309155939051.png)
**[img. Semantic segmentation의 예시]**

| ![image-20210309155947015](CV.assets/image-20210309155947015.png) |
| :----------------------------------------------------------: |
| ![image-20210309155953911](CV.assets/image-20210309155953911.png) |

**[table. 의료 사진, 자율 주행, 영상 합성 등에서 활용]**

### Semantic segmentation architectures

#### Fully Convolutional Networks(FCN)

Semantic segmentation을 위한 첫 End-to-End architecture

![image-20210309160418715](CV.assets/image-20210309160418715.png)

**[img. FCN 구조]**

- End-to-End 구조: 입력층부터 출력층까지 모두 미분가능하여 입력과 출력 pair만 있으면 모델을 학습할 수 있는 구조를 의미. 입력 사이즈 등의 제한이 없음
- 이전에는 완전학습 하기에 제한이 있었음
  - ex) AlexNet을 이용한 semantic segmentation의 경우, 학습 시의 이미지 해상도와 test 시의 이미지 해상도가 다르면 안됬음.

![image-20210309200905809](CV.assets/image-20210309200905809.png)

**[img. Fully connected layer vs Fully convolutional layer 구조 비교]**

| Fully connected layer(FCL)                                   | Fully convolutional layer                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20210309201512441](CV.assets/image-20210309201512441.png)<br />공간 정보를 고려하지 않는 모습 | ![image-20210309201522327](CV.assets/image-20210309201522327.png) |
| fixed dimensional vector를 받아 fixed dimensional vector 출력, 보통 하나로 정해진 feature vector를 출력 | activation map을 받아 activation map 출력, 보통 1x1 conv layer로 구현하며, feature vector들이 포함된 convolutional feature map 출력 |

**[table. Fully connected layer vs Fully convolutional layer 세부 비교]**

다만 Receptive field를 살핀 뒤 feature를 찾아 작은 크기의 결과를 내는 conv layer와 pooling layer로 인해 출력 이미지의 해상도가 작아짐 => 이를 해결하기 위해 Upsampling layer가 나타남

Upsampling layer이란?

![image-20210309202323525](CV.assets/image-20210309202323525.png)

**[img Upsampling이 추가된 FCN]**

작아진 결과물을 크게 만들어주기 위해 Upsampling layer을 사용

3가지 방법 중 Unpooling 방법을 제외하고 2가지 방법이 이용됨

1. Transposed convolution

![image-20210309202812424](CV.assets/image-20210309202812424.png)

**[img. Transpose convolution의 원리]**

줄어든 이미지의 픽셀을 필터만큼 곱한 뒤, kernel 사이즈와 strider 크기에 따라 곱연산하여 더한다, 중첩 되는 부분은 덧셈연산이 일어난다.

![image-20210309203617778](CV.assets/image-20210309203617778.png)

**[img. Transpose convolution의 문제점]**

kernel 사이즈와 strider 크기에 주의하지 않으면, 겹쳐서 덧셈이 일어나는 부분에 의해 checker 무늬가 나타나게 된다.

2. Upsample and convolution

위 문제를 해결하기 위해 upsampling과 convolution을 같이 사용하여 중첩하는 부분뿐만 아니라 골고루 영향을 받게 해준다.

Transpose와 달리 layer을 하나가 아닌 2개로 분리하여 주로 영상처리에 사용하는interpolation 알고리즘(Nearest-neighbor(NN), Bilinear 등)을 사용하고 convolution을 이용하여 학습 가능하게 만든다.

![image-20210309202959708](CV.assets/image-20210309202959708.png)

**[img. 개선된 convolution]**



해상도를 낮추며 진행되는 conv layer 특성상, 층의 깊에 따른 특성은 다음과 같다. 

| 낮은 레이어층, 해상도 높음, Receptive field 작음 <====> 높은 레이어층, 해상도 낮음, Receptive field 큼 |
| :----------------------------------------------------------: |
| ![image-20210309204744193](CV.assets/image-20210309204744193.png) |
| **디테일, 로컬 변화에 민감<====>전반적 의미적 정보를 포함**  |

**[table. 층의 깊이에 따른 output 값의 특징]**

결국 우리가 필요한건 구조의 깊은 부분의 의미적 부분(classify 해야하므로)과 구조의 얕은 부분의 디테일한 부분(고해상도로 픽셀을 선정해야 하므로)이 둘다 필요하므로 다음과 같은 방법으로 해결하였다.

![image-20210309211324579](CV.assets/image-20210309211324579.png)

**[img. FCN-Ns 모델들의 비교]**

마치 DenseNet이나 ResNet 처럼, 

1. 중간의 결과 값을 upsampling 한 뒤, 
2. 최종결과물을 upsampling한 것들을 
3. Concatenate하여 출력하면 좋은 결과가 나오며,

 얼마나 많은 층에서 결과값을 가져오느냐에 따라 FCN-32s, FCN-16s, FCN-8s 모델로 나누어진다.

- 숫자가 작아질 수록 더 많은 층의 결과값을 가져온 모델

![image-20210309211534377](CV.assets/image-20210309211534377.png)

**[img. FCN-Ns 모델들의 비교, 중간값을 많이 가져온 모델일 수록 정확한 결과가 나옴]**

#### Hypercolumns for object segmentation

| ![image-20210309211717755](CV.assets/image-20210309211717755.png) | ![image-20210309211729381](CV.assets/image-20210309211729381.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| HyperColumn이라는 모든 Conv layer의 결과값을 각 픽셀 별로 쌓아 만든 vecotr를 이용함 | 물체의 bounding box를 추출하고 사용한다는 점이 다름          |

**[table. FCN과 비슷한 내용을 담은 HyperColumn 논문]**

#### U-Net

영상의 일부분만 쓰는 관련된 TASK의 경우 아직도 많이 활용되는 network

Fullay convolutional network가 기반이며, skip connection을 통하여 앞선 network 보다 더욱 정교한 결과를 만들 수 있음

![image-20210309212400739](CV.assets/image-20210309212400739.png)

**[img. U자 모양이라 U-Net]**

크게 2가지 부분으로 나뉜다.

1.  Contracting path 부분

- 입력 영상을 3x3 convolution을 이용해 max pool을 이용해 해상도를 낮추고 대신 2배씩 feature channel을 늘림
- 이를 통해 전체적인 의미, 문맥(holistic context)를 확보하는 부분이며 일반적인 FCN과 다를바 없음

2. Expanding(Upsampling, decoding) path 부분

- 2x2 up-convolution을 통하여 반대로 점진적으로 채널 수는 절반으로, 해상도는 2배로 늘림

- 추가로 이전 낮은 층의 layer의 activation map을 Skip connection으로 가져와 concatenating하여 사용함

  - 이를 통해 detail하고 local한 feature map을 받아 사용할 수 있음

  - 이때 concatenate하려면 해상도가 맞아야 하는데, 홀수이면, Downsample시, 일부 값을 버리게 되며, 다시 Upsample시 해상도가 달라지므로 해상도 크기가 홀수가 안되게 해야함.
    - ex) 7x7 =DownSample(divide 2)=> 3x3 (1은 버림) =UpSample(multiple 2)=> 6x6
    - 7x7과 6x6 해상도가 맞지 않아 Concatenate 불가

**U-Net Pytorch 코드**

![image-20210309213848432](CV.assets/image-20210309213848432.png)

**[img. U-Net Contracting Path code]**

![image-20210309213905013](CV.assets/image-20210309213905013.png)

**[img. U-Net Expanding Path code]**

#### DeepLab

널리 사용되는 CRFs, Atrous Convolution의 사용이 특징인 network, Deeplab v3+가 최신.

**CRFs(Conditional Random Fields)**

후처리로 사용됨, 픽셀 간의 관계를 그래프로 표현한 뒤, 최적화하여 경계를 찾는 원리

score map과 경계선이 맞도록 경계선 내외부의 확산을 반복한다.

![image-20210309221417567](CV.assets/image-20210309221417567.png)

**[img. CRFs 예시]**

**Atrous convolution(또는 Dilated convolution)**

커널크기를 정하고, 정의한 Dilation factor 만큼 커널을 띄어 계산하는 Convolution 방법

같은 parameter 수와 연산량으로 더욱 큰 Receptive size를 얻을 수 있다.

![image-20210309221856422](CV.assets/image-20210309221856422.png)

**[img. 좌측이 기존의 conv, 우측이 astrous conv]**

**Depthwise separable convolution**

입력 이미지 해상도가 클 경우, 너무 처리가 오래 걸리자, Dilated convolution + Depthwise separable convolution = Astrous separable convolution을 이용한다.

Depthwise separable convolution는 일반 convoution을 2개의 절차로 나누어 진행한다.

![image-20210309223426502](CV.assets/image-20210309223426502.png)

**[img.Standard vs Depthwise separable convolution의 차이]**

이로 인해 파라미터 수가 $D_k^2MND_F^2$에서 $D_k^2MD_F^2+ MND_F^2$로 감소하였다.

**DeepLab v3+의 구조**

![image-20210309223952659](CV.assets/image-20210309223952659.png)

**[img. 최신 DeepLab v3+의 구조 ]**

1. DCNN 부분에서 Dilated convolution을 통하여 feature map을 구함
2. Encdoer 중간 부분에 있는 Astrous spatial pyramid pooling을 이용해 다양한 scale의 정보를 Dilated conv로 여러 feature를 추출한 후 하나로 합쳐 1x1convolution으로 하나로 합친다. 
3. Decoder 부분에서 Low-Level Features와 Upsampling한 Pyramid pooling feature를 Concat한 뒤, 결과값을 낸다.

Semantic segmentation 뿐만 아니라, instance segmentation(class 뿐만 아니라 객체 또한 탐지), panoptic segmentation(배경 정보+ instance segmentation)으로 성장하고 있다.