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