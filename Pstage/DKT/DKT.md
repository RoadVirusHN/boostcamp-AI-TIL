[TOC]

# 기계 독해(MRC)

> Naver AI boostcamp DKT 강의를 정리한 내용입니다.

## DKT 이해 및 DKT Trend 소개

### DKT Task 이해

![DKT의 원리](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524130500177.png)

DKT (DEEP KNOWLEDGE TRACING) : 딥러닝을 이용하는 지식 상태 추적

![DKT의 TASK](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524125829442.png)

Question과 Response로 이루어진 문제 풀이 정보를 통해 다음 지식상태(주로 문제를 풀 수 있는가?)를 예측하는 방식으로 진행된다.

- 즉, 주어진 문제를 맞췄는 지 틀렸는지 알아보는 Binary Classification 문제이기도 하다.

지식 상태는 계속 변화하므로 지속적으로 추적해야 한다.

보통 문제와 풀이 결과를 Train set으로,

마지막 문제의 풀이 결과가 masking 되있는 문제들과 풀이결과가 Test set으로 주어진다.

문제 풀이 정보(데이터)가 추가될 수록 학생의 지식 상태를 더 정확히 예측 가능.

데이터가 적을 수록 오버피팅 현상이 쉽게 일어난다.

### Metric 이해

#### AUC/ACC(Area under the roc curve/Accuracy)

![output과 ground-truth](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524130858408.png)

보통 예측의 결과는 float 형태로 나오며, 0.5(Threshold)를 기준으로 정답 여부(1,0)를 결정한다.

![AUC와 ACC의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524131142025.png)

#### Confusion Matrix(혼동행렬)의 이해

![Confusion Matrix의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524131556645.png)

**Predicted** : 모델의 예측값

**Actual**: 실제 값

**Accuracy**: 전체 중 예측값과 맞는 비율

**Precision(PPV, Positive predictive value)** : 모델이 맞다고 예측한 비율 중 실제 맞은 비율

**Recall,Sensitivity (True positive rate(TPR))**: 실제 1인 비율 중에 모델이 1이라고 한 비율

**Specificity** : 실제 0인 비율 중에 모델이 0이라고 한 비율

**F1 score** : Prescision과 Recall의 절충안, 동시에 고려함.

다만 위의 metric 들은 Threshold에 영향을 받게됨(여기서는 0.5)

#### AUC(Area under the roc curve)

![AUC의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524132901432.png)

그래프의 면적이 커질수록 성능이 더 좋아진다.

AUC 값의 범위는 0~1이며, 랜덤하게 0과 1을 넣은 경우 0.5이다.

AUC는 척도 불면, 절대 값이 아니라 예측이 얼마나 잘 평가되는지 측정하는 것이며(예측값들의 절대적인 크기와 관계없음),

분류 임계값 불변, 어떤 분류 임계값이 선택되었는지와 상관없이 모델의 예측 품질을 측정할 수 있다. (Threshold 관계 없음)

단, 단점들로,

척도 불변이 항상 이상적이지 않을 수 있다. 예를 들어, 0.9 이상의 값이 중요할 경우 AUC로 측정 불가

분류 임계값 불변이 항상 이상적이지 않다. 예를 들어 허위 양성(FP) 최소화가 더욱 중요한 경우(중요한 메일이 지워지면 안되는 스팸메일 분류 등) 이럴 때는 AUC가 유용한 측정항목이 아니다.

imbalanced data에서는 accuracy 보다는 낫지만, AUC가 비교적 높게 측정되는 경향이 있다.

(단, Test data가 동일할 경우, 상대적인 성능 비교는 가능하다)

![FPR vs TPR로 이루어진 AUC 그래프](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524133343483.png)

FPR은 Specificity를 의미하며, TPR은 Recall을 의미한다.

![AUC 그래프의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524133619924.png)

결과값에 따라 다음과 같은 방법으로 ROC curve를 그릴 수 있다.

![AUC Curve와 분포의 비교](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524134737191.png)

![AUC Curve와 분포의 비교](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524134808082.png)

위와 같이 Threshold 지점을 중심으로 겹치는 부분 (=예측이 틀린 부분)이 적을수록 ROC Curve의 면적이 넓어지고, 성능이 좋다는 의미이다.

### DKT History 및 Trend

ML, DL, Transformer, GNN 등의 DKT의 트랜드가 발전해 왔다.

![DKT의 트랜드](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524141306210.png)
[1강 참고 자료, History of deep knowledge tracing 참조]

## DKT Data Exploratory Data Analysis

DKT Datset EDA에 대한 예시

### i-Scream 데이터 분석

![i-Scream 데이터의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524143940935.png)

i-Scream edu에서 제공하는 Dataset

feature로 userID, assessmentItemID, testId, answerCode, Timestamp, KnowledgeTag로 이루어짐.

DKT에서 보통 하나의 행을 Interaction이라고 부름

**userID** 

- 사용자 별 고유번호, 총 7442명의 고유한 사용자 존재

**assessmentItemID**

- 사용자가 푼 문항의 일련 번호, 총 9454개의 고유한 문항이 존재
- 총 10자리로 구성, 첫자리는 항상 알파멧 A, 그다음 6자리는 시험지 번호, 마지막 3자리는 시험지 내 문항의 번호로 구성
- ex) A030071005

**testId**

- 사용자가 푼 문항이 포함된 시험지의 일련 번호, 총 1537개의 고유한 시험지가 존재
- 총 10자리로 구성, 첫 자리는 항상 알파멧 A, 그 다음 9자리 중 앞의 3자리와 끝의 3자리가 시험지 번호, 가운데 3자리는 모두 000
- 앞의 3자리 중 가운데 자리는 1~9값을 가지며 이를 대분류로 사용 가능
- ex) A030000071

**answerCode**

- 사용자가 문항을 맞았는 지 여부를 담은 이진 데이터, 0은 틀림, 1은 맞음
- 전체 Interaction에 대해 65.45%가 정답을 맞춤, 즉 조금 불균형한 데이터셋

**Timestamp**

- 사용자가 Interaction을 시작한 시간 정보, 시간 간격을 통해 문제를 푸는 시간을 가늠할 수 있음.

**KnowledgeTag**

- 문항 당 하나씩 배정되는 태그, 일종의 중분류
- 총 912개의 고유 태그 존재

#### 기술 통계량 분석

**기술 통계량?**

- 일반적으로 데이터를 살펴볼 때, 가장 먼저 살펴보는 것은 기술 통계량입니다.
- 보통 데이터 자체의 정보를 수치로 요약, 단순화하는 것을 목적으로 하며
- 우리가 잘 알고 있는 평균, 중앙값, 최대/최소와 같은 값들을 찾아내고, EDA 과정에서는 이들을 유의미하게 시각화하는 작업을 거침
- 분석은 최종 목표인 정답률과 연관 지어 진행하는 것이 유리

다음은 I-scream dataset의 특성 별 빈도 분석 종합이다.

![특성 별 빈도 분석 종합](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524145959390.png)

다음은 I-scream dataset의 특성 별 정답률 분석 종합이다.

![특성 별 정답률 분석 종합](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524150009838.png)

위와 같은 단순 기술 통계량을 넘어서, 얻어낸 특성과 정답률 사이의 관계를 분석해야 하며, 이때, 여러 지식과 경험이 있으면 좋다.

예를 들어, 문제를 많이 푼 사람이 문제를 더 잘 맞추는가?, 좀더 자주 나오는 태그의 문제의 정답률이 높은가?, 문항을 푸는데 걸린 시간과 정답률의 관계는 어떠한가? 

![푼 문항 수 vs 정답률 그래프](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524151032120.png)![평균 문항 이상 푼 학생과 이하인 학생의 정답률 분포도](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524151037369.png)

문항을 더 많이 푼 학생이 문제를 더 잘맞추는 경향이 있다.

![문항을 풀수록 한 학생의 정답률이 늘어나는 경향이 있는가?](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\DKT\DKT.assets\image-20210524151240408.png)

문항을 풀수록 한 학생의 정답률이 늘어나는 경향이 있는가?에 대한 그래프이다. 주로 초반에 잘 푼 학생은 점점 감소하며, 반대의 경우 점점 증가한다.

전반적으로 증가하는 추세이다.

이외에도 같은 시험지나 태그의 문제를 연달아 풀면 정답률이 오르는가? 등을 생각해볼 수 있다.

### Hands on EDA

[Lab. ]