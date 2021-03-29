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

## Image Classification

이미지란, 시작적 인식을 표현한 인공물(Artifact)

![이미지의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Image_Classification\Image_Classification.assets\image-20210329151150104.png)

채널이 3개인 경우 RGB(Red, Green, Blue),  채널이 4개인 겨우 CMYK(Cyan, Magenta, Yellow, Black), 또는 RGBA(+alpha)를 표현한 경우가 많다.

uint8(unsinged integeter 8bit)로 표현하는 경우가 많으며, 0~255까지 256의 범위를 가진다.

