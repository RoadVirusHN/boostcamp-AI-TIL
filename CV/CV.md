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

layer output 값을 만들기위해 Input에서 CNN layer가 참조한 공간, 클 수록 이미지의 많은 부분을 참조한 것이다.

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
3. lable된 데이터셋 +  pseudo-lable된 데이터셋을 augmentation 한 것을 통하여 새로운 Teacher model을 형성한다
   - 이때 주로 사용하는 augmentation  방법이 RandAugment
4. 새로 형성된 Teacher 모델로 2번부터 4번까지 반복한다. 

![image-20210308225809995](CV.assets/image-20210308225809995.png)

**[img. 압도적인 성능을 자랑하는 self-training 모델(빨간색)]**

