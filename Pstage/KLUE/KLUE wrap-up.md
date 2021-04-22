# KLUE wrap-up

## 성능 경과

| 번호 |                   모델                   | epoch | train loss(eval) | eval acc |  acc  | 비고                                                  |
| :--: | :--------------------------------------: | :---: | :--------------: | :------: | :---: | ----------------------------------------------------- |
|  1   |        BERT multilingual uncased         |   4   |      0.4055      |          | 70.4% | Base Model, 대소문자 구분 없음                        |
|  2   |                  kobert                  |   4   |      2.0553      |          |       | monologg/kobert                                       |
|  3   |                  kobert                  |   8   |      2.1043      |          |       | monologg/kobert                                       |
|  4   |        BERT multilingual uncased         |   8   |      0.1116      |          | 71.9% | Base Model, 대소문자 구분 없음                        |
|  5   | monologg/koelectra-base-v3-discriminator |   8   |      2.5852      |          |       | validation +label smoothing                           |
|  6   | monologg/koelectra-base-v3-discriminator |   8   |      0.1881      |          | 71.7% | validation                                            |
|  7   | monologg/koelectra-base-v3-discriminator |   8   |      0.1617      |          | 72.5% | validation 없음                                       |
|  8   |            xlm-roberta-large             |   8   |      0.1507      |          |       |                                                       |
|  9   |            xlm-roberta-large             |   8   |      2.2127      |          |       | validation                                            |
|  10  |            xlm-roberta-large             |   8   |      3.2361      |          |       | label smoothing                                       |
|  11  |            xlm-roberta-large             |  10   |  2.5864(2.8449)  |          | 77.9% | label smoothing + validation                          |
|  12  |            xlm-roberta-large             |  12   |  2.6681(2.8062)  |          |       | label smoothing + validation                          |
|  13  |            xlm-roberta-large             |  10   |  2.587(2.8190)   |          | 78.7% |                                                       |
|  14  |            xlm-roberta-large             |  10   |     ensemble     |  0.7972  |  80%  | stratifiedshuffle                                     |
|  15  |            xlm-roberta-large             |  11   |     ensemble     |          | 79.3% | eval_loss metric, low learning_rate, 8 model ensemble |

