[TOC]

# Data Visualization

데이터를 그래픽 요소로 매핑하여 시각적으로 표현하는 것

시각화의 다양한 고려 요소

1. 목적 : 왜 시각화 하는가?
2. 독자 : 시각화 결과는 누구를 위한 것인가?
3. 데이터 : 어떤 데이터를 시각화할 것인가?
4. 스토리 : 어떤 흐름으로 인사이트를 전달할 것인가?
5. 방법 : 전달하고자 하는 내용에 맞게 효과적인 방법을 사용하고 있는가?
6. 디자인 : UI에서 만족스러운 디자인을 가지고 있는가?

##  시각화의 요소

시각화를 위해서 데이터의 관점을 먼저 생각해봐야한다.

크게 1. 데이터셋 관점 (global)과

2. 개별 데이터의 관점 (local) 관점이 존재한다.

### 데이터셋, 데이터의 종류

정형 데이터, 시계열 데이터, 지리 데이터, 관계형(네트워크) 데이터, 계층적 데이터, 다양한 비정형 데이터 등이 존재한다.

![정형 데이터](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210405120038345.png)

이때, 정형 데이터는 테이블 형태로 제공되는 데이터로 csv, tsv 파일 등으로 제공됨.

Row가 데이터 1개 item, Column은 attribute(feature)를 의미하며, 시각화가 비교적 쉽다.

![시계열 데이터](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210405120144753.png)

시간 흐름에 따른 데이터를 Time-Series 데이터

기온, 주가 등의 정형 데이터와 음성, 비디오 같은 비정형 데이터가 존재

추세(Trend), 계절성(Seasonality), 주기성(Cycle) 등을 살핌

![지리/지도 데이터](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210405120235991.png)

지리/ 지도 데이터의 경우, 지도 정보와 보고자 하는 정보간의 조화 중요 + 지도 정보를 단순화 시킨 경우가 중요하며, 거리 경로, 분포 등을 Visualization

![관계 데이터](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210405120345446.png)

관계 데이터(Graph, Network Visualization)의 경우 객체와 객체 간의 관계를 시각화,

객체를 Node, 관계를 Link라고 하며, 크기, 색, 수 등으로 객체와 관계의 가중치 표현

![계층적 데이터](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210405120504372.png) 

계층적 데이터의 경우, 관계 중에서도 포함관계가 분명한 데이터, Tree, Treemap, Sunburst 등이 대표적



데이터의 종류는 4가지로 분류 가능

수치형(numerical): 수로 표현 가능한 데이터

- 연속형(continuous) : 실수값으로 표현 가능한 연속적인 데이터
  - 길이, 무게, 온도 등
- 이산형(discrete) : 정수값으로 표현 가능한 데이터
  - 주사위 눈금, 사람 수 등

범주형(categorical): class로 표현 가능한 데이터

- 명목형(norminal) : 순서나 대소 관계, 우열관계가 존재하지 않음
  - 혈액형, 종교 등
- 순서형(ordinal) : 순서, 대소관계, 우열 관계가 존재
  - 학년, 별점, 등급 등

### 시각화의 이해

![mark의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210405121229051.png)

Mark는 기본적인 시각적 요소로, 점, 선, 면으로 이루어진 시각화 방법이다.

질의 데이터, 시계열 데이터 등이 표현 가능

![채널의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210405121257051.png)

위의 기본적인 점선면 Mark에 다양성을 줄 수 있는 요소이다, 크기, shape, color 등을 바꾸어 다차원적인 요소를 표현 가능하다.

![전주의적 속성](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210405121312065.png)

전주의적 속성(Pre-attentive Attribute)는 시작적인 주의를 주지 않아도 인지하게 되는 요소를 의미한다.

동시에 사용하면 인지하기 어려우므로 적절하게 사용할 때, 시각적 분리(visual pop-out)이 필요하다.

## 기본 Bar Plot

Bar plot이란 직사각형 막대를 사용하여 데이터의 값을 표현하는 차트/그래프를 의미하며, 범주(category)에 따른 수치 값을 개별 또는 그룹 별로 비교하기에 적합하다.

막대의 방향에 따라 **.bar()  / .barh()**으로 수직과 수평형 그래프를 만들 수 있다.

- 수평은 범주가 많을 때 적합하다.

![bar vs barh](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210425225339626.png)

### 여러 Bar plot

범주형 A, B, C, D, E column을 가진 두 그룹 파랑 = [1,2,3,4,3], 분홍 = [4,3,2,5,1]이 있을 때,

1. **플롯을 여러 개 그리는 방법**

![2개의 플롯으로 표현](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210425232444487.png)

2. **한 개의 플롯에 동시에 나타내는 방법**

![Stacked Bar plot](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210425232839799.png)

1)  Stacked Bar Plot : 위에 쌓여있는 bar의 분포는 파악하기 쉬움, 2개의 그룹이면 y axis = 0를 중심으로 +, - 로 축조절로 극복 가능

- .bar()에서는 bottom 파라미터로 사용, .barh()에서는 left 파라미터로 사용

![Percentage Stacked Bar Chart](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210425233005642.png)

1-1) Percentage Stacked Bar Chart : Stacked Bar Plot의 응용 형태

![Overlapped Bar Plot](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210425233142292.png)

2) Overlapped Bar Plot : 3개 미만의 급룸 비교는 겹쳐서 표현 가능, 투명도(alpha)를 조정해 겹치는 부분 파악, Area plot에 효과적



![Grouped Bar Plot](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210425233447775.png)

3) Grouped Bar Plot: 구현이 까다로움, 그룹이 5~7개 이하일때 효과적 그 이상은 etc로 처리할 것

- (.set_xticks(), .set_xticklabels())로 구현

### 정확한 Bar plot

![잘못된 비례 관계 그래프](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210425233831294.png)

x축의 시작은 zero(0)부터이며, 실제 값과 그래픽으로 표현 되는 잉크양은 비례해야 한다.

![정렬을 통해 패턴이 보이는 그래프](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210425234257251.png)

데이터의 정렬을 통해 데이터의 패턴을 발견할 수 있다.

시게열(시간순), 수치형(크기순), 순서형(범주의 순서), 명목형(범주의 값)에 따라 정렬이 가능하며, 대시보드에서는 Interactive 하게 제공하는 것이 유용하다.

pandas에서는 sort_values(), sort_index()를 사용하여 정렬 가능

![여백에 따른 가독성 차이](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210425234744834.png)

여백과 공간을 조정해서 가독성을 높일 수 있다

- X/Y axis Limit (.xset_xlim(), .set_ylim())

- Spines(.spines[spine].set_visible())
- Gap(width)
- Legend(.legend())
- Margins.(.margins())

등으로 조절 가능하다.

![3d 그래프](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210425235714942.png)

무의미한 3D, 직사각형 의외의 bar 형태 지양

- Grid(.grid())

- Ticklabels(.set_ticklabels())
  - Major & Minor

- Text 추가 장소 (.text(), .annotate())
  - Bar의 middle/ upper
  - 제목 (.set_titile())
  - 라벨 (.set_xlabel(), .set_ylabel())

오차 막대 (error bar)로 Uncertainty 정보 추가 가능

Bar 사이의 Gap이 없으면 히스토그램(Histogram)이 되며, 연속된 느낌이며, .hist()를 통해 사용 가능

 