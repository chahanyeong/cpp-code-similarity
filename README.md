# 대회 소개
:link: [코드 유사성 판단 시즌2 AI 경진대회](https://dacon.io/competitions/official/236228/overview/description)

# 팀이 바라본 대회의 핵심
1. 모델 Input인 C++ 코드파일의 최적화 클리닝
   - C++코드에는 '#include', '#define', 'using' 등 다양한 전처리기가 존재한다.
   - 이런 다양한 전처리기와 주석을 어떻게 제거 및 치환할지 특히 집중했다.

2. 모델 훈련 데이터를 얼마나 증강할 수 있는가
   - 비슷한 기존 대회와 비교하였을때 대체로 훈련데이터의 개수와 정확도는 비례하였다.
   - 정해진 리소스 안에서 최대의 정확도를 내기위한 방법에 대해 연구했다.

# 분석 과정
![모델링 과정](https://github.com/chahanyeong/cpp-code-similarity/assets/152364900/513b5dcc-5f71-40a1-b384-53f8dc91a957)

# 디렉토리
```bash
├── data
│   ├── new_dataset_0604
│   ├── new_dataset_0607
│   ├── graphcodebert
│   └── codebert-cpp
├── code
├── code_submission_total.py
├── code_submission_ensemble.py
├── train_code
│   │
│   └── probelm 1 ~ 500
│
├── test.csv
└── sample_submission.csv
``` 

### Data Cleaning
![코드 전처리기 예시](https://github.com/chahanyeong/cpp-code-similarity/assets/152364900/cb910f9b-1ad6-4cd7-b351-db370dba1d2b)
- 다양한 전처리기 구문 처리
  - '#include', 'tie', 'ios' 등은 단순 제거 시 분류 정확도를 높혀준다.
  - '#define', '#typedef', 'using' 등은 코드 구문을 치환하는 역할을 하므로 단순제거 vs 정규연산자를 활용한 구문 치환으로 Score를 비교했다.(20000개 train sample 기준)
  - Method 1 채택

|               Cleaning                | Method | Public Score |
| :-------------------------------: | :----: | :--: |
|     전처리기 구문 단순 제거      |  1  | 93.54 |
|          제거 및 치환          |  2  | 92.26 |


### Model Selecting
- 모델 비교 결과 neulab/codebert-cpp, microsoft/graphcodebert-base 채택했다.(20000개 train sample 기준)
- 위의 두모델은 훌륭하게 기학습되었으나 토크나이저가 비효율적(데이터셋을 지나치게 잘개 쪼갬 -> 데이터의 20% 이상이최대제한길이인 512토큰을 넘어섬)
- PLbart-large의 경우 코드 토크나이징 시 90%의 데이터가 512토큰안에 들어갔지만 모델이 지나치게 커 제한시간안에 사용 못할것이라 판단했다.


|               Model                | Public Score |
| :-------------------------------: | :--: |
|     neulab/codebert-cpp     | 93.54 |
|           microsoft/graphcodebert-base          | 92.26 |
|          mrm8488/CodeBERTaPy          | 59.74 |
|          uclanlp/plbart-large          | - |


### Create Data
- Colab A100 기준 최대 훈련 데이터 개수는 180만 행 이였다.(Colab pro+ 기준 셀 지속시간: 24시간)
- 생성 가능한 데이터 행은 약 1억 2000개 였고 이 중 효율적인 Data sampling을 위해 두가지 방식을 비교하였다.
- Random sampling 채택
  - Positive pair -> Random sampling
  - Negative pair -> BM25plus 알고리즘 사용한 구성 vs Random sampling
'BM25plus' 알고리즘은 키워드 기반의 랭킹 알고리즘으로 두 코드간의 유사성을 Score값으로 나타내준다. Negative pair를 구성시 해당 알고리즘을 사용해 가장 비슷하면서 결과적으로 유사하지 않은 코드쌍을 추출하는 방식을 사용하였다. 하지만 기존에 사용하는 데이터의 개수가 180만개로 비교적 적었기 때문에 오히려 유사한 경우를 유사하지 않다고 판단해 Random sampling보다 낮은 정확도를 보인것으로 추측된다.

|    Sampling Method(Graphcodebert)    | Public Score |
| :-------------------------------: | :--: |
|     BM25plus     | 97.52 |
|     Random Sampling    | 97.74 |

### Fine-Tuning, Ensemble
- neulab/codebert-cpp, microsoft/graphcodebert-base 각 180만 행으로 fine-tuning 완료.
- 각 2Epoch, batch: train32 test4
- 두 모델의 Prediction을 Hard Voting 해 Public Score 98.475 달성
|    Model    | Public Score |
| :-------------------------------: | :--: |
|     neulab/codebert-cpp     | 97.94 |
|     microsoft/graphcodebert-base    | 98.15 |
|     Ensemble-Hard Voting    | 98.475 |


# 팀원

- 팀장 : 차한영 (AI 엔지니어)
- 팀원 : 김태균 (수학과, AI융합전공 학부생)
