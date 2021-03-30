[TOC]

# Image Classification Competition

> 네이버 AI BoostCamp의 P stage 이미지 분류 강의를 정리하였습니다.

 ## Competition with AI Stages

**Competition이란?**

![kaggle 로고](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210329121207833.png)![DACON 로고](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210329121158502.png)

여러 회사, 단체 등에서 상금을 내걸고 데이터를 제공하여 원하는 결과를 얻는 가장 좋은 방법을 모색한다.

- 데이터 뿐만 아니라 컴퓨팅 자원을 지원할 때도 있다.

Kaggle이 competition platform으로 대표적이며, 국내에서는 DACON 이 존재한다.

|                      |                           Pipeline                           |
| :------------------: | :----------------------------------------------------------: |
|     ML pipeline      | ![image-20210329123936670](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210329123936670.png) |
| Competition pipeline | ![image-20210329123944543](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210329123944543.png) |

실제 머신러닝 업무의 Pipeline과 Competition의 Pipeline이 비슷한 경우가 많기 때문에 Competition을 통해 ML 실력을 갈고 닦기 좋다!

**Overview**

![Overview의 대표적인 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210329121716649.png)

문제에 대한 정의, 요구사항과 직면하고 있는 문제점 등에 대해서 자세히 알 수 있다.

위의 경우, 단순히 악성 댓글을 파악하는 문제가 아니라, 악성댓글 파악 중 야기되는 정치적 이슈에 대한 설명이 포함되어 있다.

위의 overview를 자세히 보는 것으로 내가 풀어나가야할 문제에 대한 방향성을 얻어낼 수 있으며, 이러한 행위를 Problem Definition(문제 정의)라고 한다.

이러한 문제 정의 행위로는

1. 내가 무슨 문제를 풀어야 하는가?
2. Input과 Output은 무엇인가?
3. 이 솔루션은 어디서 어떻게 사용되어지는가?

등의 예시가 있다.

**Data Description**

![(Data Description의 예시(https://www.kaggle.com/ranzcr-clip-catheter-line-classification/data)](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210329122428314.png)

File 형태, Metadata Field 소개 및 설명이 적혀있는 부분

데이터 스팩의 요약본, EDA, 문제정의 등에 중요하다.

**Notebook**

![Jupyter notebook](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210329122636119.png)

데이터 분석, 모델 학습, 테스트 셋 추론의 과정을 서버에서 연습 가능

단순히 연습용이 아니라, 일부 Competition의 경우는 회사에서 이러한 방식으로 Computing power를 제공하는 경우도 있으며, 성능을 강제하기 위해 사용이 강요되는 경우도 있다.

**Submission**

![나의 제출물 확인](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210329122950431.png)

자신이 제출했던 테스트 예측 결과물들의 정보를 확인하고, 제출할 결과물을 선택할 수 있다.

**Leaderboard**

![나의 순위 확인](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210329123023255.png)

자신의 제출물의 순위를 확인할 수 있다.

**Discussion**

![https://www.kaggle.com/c/stanford-covid-vaccine](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210329123057805.png)

Kaggle 등에서는 Discussion에서 다양한 정보를 주고 받는 문화가 발달함.

정보를 공유함으로, 경쟁자의 점수가 올라갈 수 있음에도 불구하고, 등수를 올리는 것 보단 문제를 해결하고 싶은 마음으로 올리는 경우가 많음.

하지만 경연 종료 1~2주 전에는 보통 Critical한 정보의 공유는 멈춰지는 경우가 많다.

ML에 관한 정보뿐만 아니라, 도메인 지식, CS 지식 등도 공유된다.

## EDA(Exploratory Data Analysis, 탐색적 데이터 분석)

![image-20210329144512799](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210329144512799.png)

EDA는 ML pipeline 중 Data Analysis 부분에 해당한다

### EDA란?

EDA(Exploratory Data Analysis, 탐색적 데이터 분석), 데이터를 이해하기 위한 노력

데이터의 구성, 형태, 쓰임새 등을 분석하는 것

- 주제와 연관성, 데이터 타입의 특성, 메타데이터의 분포 등이 있다.

머신러닝 파이프라인에 많은 것에 영향을 주게 된다.

![https://www.kaggle.com/ash316/eda-to-prediction-dietanic](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210329142136911.png)

EDA 결과물의 예시, 데이터 이해가 목적이므로 정해진 툴이나 절차 없이 궁금한 점을 찾아보는 과정이다.

또한, 한번으로 끝나는 과정이 아니며, 언제나 의문이 생기면 EDA는 pipeline 과정 중간에 언제나 돌아올 수 있다.

툴로는 python, excel 등이 존재

### Image Classification

이미지란, 시작적 인식을 표현한 인공물(Artifact)

![이미지의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210329151150104.png)

채널이 3개인 경우 RGB(Red, Green, Blue),  채널이 4개인 겨우 CMYK(Cyan, Magenta, Yellow, Black), 또는 RGBA(+alpha)를 표현한 경우가 많다.

uint8(unsinged integeter 8bit)로 표현하는 경우가 많으며, 0~255까지 256의 범위를 가진다.

## Dataset

 ![DataPipeline](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210330100946006.png)

Dataset 형성, Data Generation은 Data Processing의 과정이다.

Raw Data를 모델이 사용할 수 있는 Dataset으로 만들면 다양한 모델이나 쓰임새에 쉽게 사용할 수 있을 뿐만 아니라 인간이 쉽게 이해할 수 있다.

### Pre-processing

Pre-processing(전처리)는 pipeline에 시간과 중요성, 비중 모두 커다란 부분을 차지하고 있다.

![Pre-processing의 비중](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210330101722733.png)

Competition data는 제대로된 Data를 주는 편이지만, 수집한 데이터의 경우 절반 이상을 날려야 하는 경우가 많음.

이미지의 전처리의 경우, 다음과 같은 예시가 있다.

- Bounding box

![Bounding box의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210330101926520.png)

이미지의 중요한 부분을 Crop한 뒤 Resize하여 학습시킬 모델을 더욱 명확히 함

좌측 상단과 우측 하단의 좌표를 이용해 Bounding box를 그릴 수 있음

- Resize

![데이터 Resize](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210330101939193.png)

 계산효율을 위해 적당한 크기로 이미지 사이즈 변경

이미지 사이즈가 바뀌어도 성능에 영향은 미미한 경우가 많다.

- 이미지 채도, 대비 변경

![APTOS Blindness Detection](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210330102840559.png)

상단은 홍채 사진을 더욱 뚜렷히 바꾼 것

명확하지 않은 사진을 채도, 대비, 명도,등을 바꾸어 선명하게 바꿈

전처리가 성능의 향상을 보장하진 않는다.

## Generalization

### Bias & Variance

![Underfitting과 Overfitting의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210330103626980.png)

Underfitting : 학습이 부족하여 제대로 구별할 수 없음

Overfitting : noise를 포함해서 모든 학습이 너무 많이되어 너무 자세하게 구분함

일반화, 즉 학습한 데이터셋 이외의 Input에서는 부정확한 결과를 내게 됨

### Train/Validation

![Validation Set](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210330105057803.png)

 훈련 셋 중 일정 부분을 따로 분리하여 검증 셋으로 활용

학습 가능한 데이터양이 줄어들지만, 학습되지 않은 Validation Set의 분포를 통해 일반화 성능을 검증할 수 있다.

검증 절차를 통해 Generalization(일반화) 성능을 늘릴 수 있다.

### Data Augmentation

![Data Augmentation의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210330105400221.png)

데이터를 일반화하는 과정, 주어진 데이터가 가질 수 있는 Case(경우), State(상태)를 다양하게 함.

이미지의 경우 다양한 필터, 채도, 색상 변경, 이미지 회전, Crop 등, 원본 이미지를 변경하는 것을 통하여 데이터를 늘림.

문제가 만들어진 배경과 모델의 쓰임새를 살표보아 Data Augmentation의 힌트를 얻을 수 있다.

(ex)밤과 낮에 따로 촬영될 수 있는가? -> 명도 변경)

torchvision.transforms 에서 Image에 적용 가능한 다양한 함수를 살펴볼 수 있다.

 ![image-20210330110154448](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210330110154448.png)

![torchvision.transforms 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210330110312284.png)

Albumentations라는 Library는 좀 더 처리가 빠르고 다양한 기능을 제공함

![Albumentations 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210330110449254.png)

## Data Generation

데이터 전처리, Augmentation 등의 출력 결과가 모델의 성능, 시간 등에 영향을 주는 경우가 많다.

### Data Feeding

![image-20210330114004455](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210330114004455.png)



모델의 학습속도와 이전 단계가 느리다면 전체적인 학습속도가 느려질 수 밖에 없다.

![데이터셋 생성 능력 비교](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210330114020650.png)

단순히 Data Augmentation의 순서가 달라지는 것 많으로도 성능이 달라진다.

데이터셋 생성능력을 비교하여 적절한 성능을 정해보자.

### torch.utils.data

**Dataset**

pytorch에서 정의할 수 있는 데이터셋 생성 방법은 다음과 같다.

```python
from torch.utils.data import Dataset

class MyDataset(Dataset): # torch.utils.data.Dataset 상속
    def __init__(self): # MyDataset 클래스가 처음 선언되었을 때 호출
        print("Class init!!")
        pass
    # test = MyDataset() 결과: # Class init!!
   	
    def __getitem__(self, index): # MyDataset의 데이터 중 index 위치의 아이템을 리턴
        return index
   	# test[2341] 결과: # 2341
    
	def __len__(self):  # MyDataset 아이템의 전체 길이
        print("length is 3")
        return 3
    # len(test) 결과: # lenth is 3 # 3
```

이를 통해 Vanilla 데이터를 원하는 형태로 출력 가능

**DataLoader**

Dataset을 효율적으로 사용할 수 있도록 관련 기능 추가

```python
train_loader = torch.utils.data.DataLodaer(
train_set,
batch_size=batch_size, # batch 사이즈
num_workers=num_workers, # 병렬 코어 갯수
drop_last=True # batch 사이즈에 맞지 않는 마지막 batch 버림
)
```

이외에도 shufle, sampler, batch_sampler, pin_memory 등 여러 기능이 있다.

collate_fn : batch 단위 마다 실행할 함수 넣어주기

![num_workers 사용 효과](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210330133541405.png)

num_workers 사용 결과

너무 큰 값을 사용하면, 딥러닝 이외의 시스템에서의 사용하는 연산이 간섭하여 오히려 성능이 떨어진다고 함.

