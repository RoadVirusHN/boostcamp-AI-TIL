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
## 기본적인 차트의 사용
### 기본 Bar Plot

Bar plot이란 직사각형 막대를 사용하여 데이터의 값을 표현하는 차트/그래프를 의미하며, 범주(category)에 따른 수치 값을 개별 또는 그룹 별로 비교하기에 적합하다.

막대의 방향에 따라 **.bar()  / .barh()**으로 수직과 수평형 그래프를 만들 수 있다.

- 수평은 범주가 많을 때 적합하다.

![bar vs barh](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210425225339626.png)

#### 여러 Bar plot

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

#### 정확한 Bar plot

![잘못된 비례 관계 그래프](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210425233831294.png)

Principle of Proportion Ink : x축의 시작은 zero(0)부터이며, 실제 값과 그래픽으로 표현 되는 잉크양은 비례해야 한다.

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

### Line Plot

![Line Plot의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426001044039.png)

연속적으로 변화하는 값을 순서대로 점으로 나타내고, 이를 선으로 연결한 그래프
#### Line Plot의 기본
Line plot은 꺾은선 그래프, 선 그래프, line chart, line graph 등의 이름으로도 불리며, 시간/순서에 대한 변확에 적합하여 추세(시계열 분석)를 살피기 위해 사용함.(.plot()으로 사용)



가독성을 위해 5개 이하의 선을 여러 요소로 구별하며 사용하는 것이 좋다.

1. 색상(color)
2. 마커(marker, markersize)
3. 선의 종류(linestyle, linewidhth)

![구별 가능한 선의 종류](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426001552666.png)

또한, Noise로 인하여 패턴 및 추세 파악이 힘들 경우 smoothing을 통해 가독성을 늘릴 수 있다.

![smoothing을 통한 패턴 파악](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426001821085.png)

#### 정확한 Line plot

![추세에 집중된 Line plot](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426003104044.png)

1. **추세에 집중하기**

- 추세가 목적이므로 굳이 0을 시작점으로 두지 않아도 된다.

- Grid, Annotate 등을 제거한 뒤, 디테일한 정보는 표로 제공하자.

- 생략되지 않는 성에서 범위를 조정해 변화율 관찰 (.set_ylim())

![image-20210426003320621](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426003320621.png)

2. **간격 조정**

- 규칙적인 간격이 아닐 경우 점을 추가하여 데이터가 있는 부분만 표시하자.

3. **보간**

- Line은 점을 이어 만드는 요소이므로 데이터가 없어도 이를 이어서 만드는 보간을 하게 된다.
- 데이터의 이해를 도울 수 있지만, 없는 데이터를 있다고 생각하거나 적은 차이를 못보게 할 수 있으므로 일반적인 분석에서는 지향 하자.

- 데이터의 error나 noise가 포함되어 있는 경우, Moving Average 방법, Smooth Curve with Scipy(scipy의 interpolate 내부의 make_interp_spline(), interp1d() 또는 scipy의 ndimage.gaussian_filter1d() 등을 사용 가능)

![이중 축 그래프의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426004152188.png)

4. **이중 축 사용**

![이중축 vs 다중 plot의 가독성 차이](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426004221912.png)

- 한 plot에 대해 2개의 축을 이중 축(dual axis)라고 함, 왠만하면 이중 축보단 다중 plot으로 해결하는 게 좋다.
- 같은 시간 축에 대해 서로 다른 종류의 데이터를 표현하기 위해 축이 2개 필요 (.twinx() 사용)
- 한 데이터에 대해 단위가 다른 경우, .secondary_xaxis(), .secondary_yaxis()를 사용해 보자.

5. **기타**

![Line 끝 단에 레이블 추가](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426004318619.png)

- 범례 대신 라인 주위에 레이블을 추가하면 가독성이 좋다.

![Min/Max Info가 추가된 그래프](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426004419061.png)

- Min/Max 정보(또는 특정 포인트)를 추가(annotation)로 가독성 증가

![신뢰구간의 표현](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426004517987.png)

- 보다 연한 색을 사용하여 uncertainty 표현 가능 (신뢰구간, 분산 등)

### Scatter Plot

![Basic Scatter Plot](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426005150349.png)

#### 기본 Scatter plot

![Scatter plot의 요소](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426010247769.png)

점을 사용하여 두 feature 간의 관계를 알기 위해 사용하는 그래프, 산점도라고도 말함

직교 좌표계에서 x축/y축에 feature r값을 매핑해서 사용, .scatter()로 사용

색, 모양, 크기 등을 바꾸어 다양한 차원의 데이터 표현 가능

![image-20210426010716264](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426010716264.png)

![군집, 값의 차이, 이상치](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426010727232.png)

Scatter plot을 통해 상관관계(양, 음의 상관 관계, 상관 없음), 군집, 값 사이의 차이, 이상치 등을 알 수 있다.



#### 정확한 Scatter plot

![점의 분포 파악을 위해 변환 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426011708647.png)

1. **Overplotting 방지**

- 점이 너무 많으면 분포 파악이 힘드므로, 투명도 조정, 지터링(jittering, 점의 위치 약간 변경), 2차원 히스토그램(히트맵을 사용하여 깔끔한 시각화), Contour plot(분포를 등고선을 사용해 포현) 등으로 표현하면 좋다.





![점의 요소와 인지](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426012238441.png)

2. **점의 요소와 인지**

- 색 : 연속은 gradient, 이산은 개별 색상으로 표시
- 마커 : 거의 구별하기 힘들고, 크기가 달라 보이므로 사용 유의
- 크기 : 흔히 버블 차트(bubble chart)라고 부름, 오용이 잦으며, 관계 보다는 각 점간의 비율에 초점을 두자, SWOT 분석등에 자주 사용



3. **인과 관계와 상관 관계**

- 인과 관계(causal relation)과 상관 관계(correlation)을 잘 구별하자
- 인과 관계는 사전 정보와 함께 가정으로 제시





![추세선의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\Data_Visualization\Data_Visualization.assets\image-20210426012304744.png)

4. **추세선**

- 추세선을 이용하면 scatter의 패턴을 유추할 수 있지만, 2개 이상 사용하면 가독성이 떨어진다.

5. 기타

- Grid는 왠만하면 사용하지 않으며 무채색을 활용하자
- 범주형이 포함되어 있으면 heatmap 또는 bubble chart 추천




## 차트의 요소

### Text 사용하기

#### Matplotlib에서 Text



#### 실습



### Color 사용하기

#### Color에 대한 이해



#### Color Palette의 종류



#### 그 외 색 Tips



### Facet

#### 기본 Facet



#### Matplotlib에서 구현




### More Tips

#### Grid 이해하기



#### 심플한 처리



#### Setting 바꾸기

