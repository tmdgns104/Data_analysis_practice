# 필요한 라이브러리 임포트
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import polars as pl
import gc
import pickle
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)  # 모든 컬럼 보기

# 머신러닝 모델
import lightgbm as lgb

# 교차 검증을 위한 KFold
from sklearn.model_selection import KFold

# 화학 구조 분석을 위한 라이브러리
#rdkit은 파이참에서 설치가 쉽지 않아 포기
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdmolops

# 설정 클래스 정의
class CFG:
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']  # 예측 대상 Tg: 유리전이온도 FFV: 자유부피비 Tc: 임계온도 Density: 밀도 Rg: 반지름
    SEED = 42 #무작위 값을 생성할 때 기준이 되는 숫자 42는 관행적 농담
    FOLDS = 5 #K-folds를 할때 몇등분으로 나눌지
    PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'  # 데이터 경로

# 학습/테스트 데이터 로드
train = pd.read_csv(CFG.PATH + 'train.csv')
test = pd.read_csv(CFG.PATH + 'test.csv')

# SMILES를 canonical form으로 변환
def make_smile_canonical(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        #rdkit의 MolFromSmiles로 SMILES를 분자 객체 (mol)로 변환.
        return Chem.MolToSmiles(mol, canonical=True)
        #변환한 분자 객체를 다시 문자열로 바꾸되, canonical=True 옵션을 통해 정렬된(일관된) 형태로 변환.
    except:
        return np.nan
        #SMILES 파싱이 실패하면 결측치

train['SMILES'] = train['SMILES'].apply(make_smile_canonical)
test['SMILES'] = test['SMILES'].apply(make_smile_canonical)
#SMILES (Simplified Molecular Input Line Entry System)**는 분자를 문자열로 표현하는 방식
#같은 분자라도 여러 방식으로 표현할 수 있어서, "표준 표현"으로 변환해야 중복을 피하고 일관된 분석



# 외부 데이터 로드 및 컬럼명 수정
#train.csv를 보완하기 위한 추가적인 학습 데이터 소스
#train 데이터에 병합되거나 결측 보완에 활용

# 1. Tc 데이터
data_tc = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')
data_tc = data_tc.rename(columns={'TC_mean': 'Tc'})
#train 데이터와 컬러명 맞추기 TC_mean -> TC

# 2. Tg 데이터 1
data_tg2 = pd.read_csv('/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv', usecols=['SMILES', 'Tg (C)'])
data_tg2 = data_tg2.rename(columns={'Tg (C)': 'Tg'})
#train 데이터와 컬러명 맞추기 Tg (C) -> Tg

# 3. Tg 데이터 2 (K -> C 변환)
data_tg3 = pd.read_excel('/kaggle/input/smiles-extra-data/data_tg3.xlsx')
data_tg3 = data_tg3.rename(columns={'Tg [K]': 'Tg'})
data_tg3['Tg'] = data_tg3['Tg'] - 273.15
#train 데이터와 컬러명 맞추기 Tg [K] -> Tg
#캘빈에서 도씨로 변경

# 4. 밀도 데이터 (float 변환 및 전처리)
data_dnst = pd.read_excel('/kaggle/input/smiles-extra-data/data_dnst1.xlsx')
data_dnst = data_dnst.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
#밀도만 가져오고 컬럼 같게 맞추기
data_dnst['SMILES'] = data_dnst['SMILES'].apply(make_smile_canonical)
#SMILES 정규화
data_dnst = data_dnst[
    (data_dnst['SMILES'].notnull()) & (data_dnst['Density'].notnull()) & (data_dnst['Density'] != 'nylon')]
#밀도중에 SMILES를가 결측치가 없고 밀도가 결측치가없고 밀도가 nylon이 아닌것 만 남김
# "nylon"은 잘못된 입력(수치가 아닌 텍스트)으로 간주하고 제거
data_dnst['Density'] = data_dnst['Density'].astype('float64')
#문자열/혼합 타입으로 되어 있을 수 있는 Density 값을 float64로 변환
data_dnst['Density'] -= 0.118
#Density 값에서 0.118을 일괄적으로 빼는 보정 작업
#이건 원 데이터가 다른 실험 조건 하에서 측정되었기 때문
#데이터 정규화/보정(normalization) 차원에서 오프셋을 적용한거 같음 왜인지 자세히 모름

# 외부 데이터를 합치는 함수 정의
# 기존 train에 없는 SMILES는 새 행으로 추가
# 이미 있는 SMILES인데 target 값이 없는 경우는 보충
# 이미 있는 SMILES인데 target 값도 있으면 덮어쓰지 않음
# df_train: 원래 학습 데이터
# df_extra: 외부 추가 데이터셋
# target: 예측할 물성 중 하나
def add_extra_data(df_train, df_extra, target):
    n_samples_before = len(df_train[df_train[target].notnull()])
    #기존 학습 데이터에서 해당 target 값이 있는 sample 수 확인
    df_extra['SMILES'] = df_extra['SMILES'].apply(make_smile_canonical)
    #SMILES를 canonical form으로 정리한 후
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    #SMILES가 중복일 경우, target 값을 평균내어 하나로 정리
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])  # 외부 내부 둘 다 존재하는 SMILES
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES']) # 외부에만 존재하는 SMILES


    #중복되더라도 train에 값이 있는 경우는 유지
    #값이 없는 경우에만 외부 데이터를 써서 보완함
    for smile in df_train[df_train[target].notnull()]['SMILES'].tolist():
        #smile에 트래인 내부데이터중 타겟값이 있는 곳의 SMILES를 리스트 형태로 불러와서 하나하나 넣어줌
        if smile in cross_smiles:
            #불러온 것중 내부외부가 겹치면는 거면
            cross_smiles.remove(smile)
            #cross_smiles에서 제거

    for smile in cross_smiles:
        #내부외부동시에 있는 smiles중 타겟값이 없는것만 남음
        df_train.loc[df_train['SMILES'] == smile, target] = df_extra[df_extra['SMILES'] == smile][target].values[0]
        #내부데이터[타겟값이 없는것,물성] 에 외부데이터[타겟값이 없는것] 의 타겟값(리스트형태로 나오는거 같은데 그중 첫번째)

    df_train = pd.concat([df_train, df_extra[df_extra['SMILES'].isin(unique_smiles_extra)]], axis=0).reset_index(drop=True)
    #df_train = 내부데이터 + 외부데이터중 외부에만 있는거 , 열방향으로 합침(위아래로 붙임),(데이터프레임의 인덱스를 초기화,기존 인덱스를 버림)


    n_samples_after = len(df_train[df_train[target].notnull()])
    # 학습 데이터에서 해당 target 값이 있는 sample 수를 다시 확인
    print(f'\nFor target "{target}" added {n_samples_after - n_samples_before} new samples!')
    return df_train

# 외부 데이터를 train에 병합
train = add_extra_data(train, data_tc, 'Tc')
train = add_extra_data(train, data_tg2, 'Tg')
train = add_extra_data(train, data_tg3, 'Tg')
train = add_extra_data(train, data_dnst, 'Density')

# 불필요하거나 상관이 너무 높은 feature 제거용 리스트 정의
useless_cols = [
    # Nan data
    'BCUT2D_MWHI',
    'BCUT2D_MWLOW',
    'BCUT2D_CHGHI',
    'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI',
    'BCUT2D_LOGPLOW',
    'BCUT2D_MRHI',
    'BCUT2D_MRLOW',

    # Constant data
    'NumRadicalElectrons',
    'SMR_VSA8',
    'SlogP_VSA9',
    'fr_barbitur',
    'fr_benzodiazepine',
    'fr_dihydropyridine',
    'fr_epoxide',
    'fr_isothiocyan',
    'fr_lactam',
    'fr_nitroso',
    'fr_prisulfonamd',
    'fr_thiocyan',

    # High correlated data >0.95
    'MaxEStateIndex',
    'HeavyAtomMolWt',
    'ExactMolWt',
    'NumValenceElectrons',
    'Chi0',
    'Chi0n',
    'Chi0v',
    'Chi1',
    'Chi1n',
    'Chi1v',
    'Chi2n',
    'Kappa1',
    'LabuteASA',
    'HeavyAtomCount',
    'MolMR',
    'Chi3n',
    'BertzCT',
    'Chi2v',
    'Chi4n',
    'HallKierAlpha',
    'Chi3v',
    'Chi4v',
    'MinAbsPartialCharge',
    'MinPartialCharge',
    'MaxAbsPartialCharge',
    'FpDensityMorgan2',
    'FpDensityMorgan3',
    'Phi',
    'Kappa3',
    'fr_nitrile',
    'SlogP_VSA6',
    'NumAromaticCarbocycles',
    'NumAromaticRings',
    'fr_benzene',
    'VSA_EState6',
    'NOCount',
    'fr_C_O',
    'fr_C_O_noCOO',
    'NumHDonors',
    'fr_amide',
    'fr_Nhpyrrole',
    'fr_phenol',
    'fr_phenol_noOrthoHbond',
    'fr_COO2',
    'fr_halogen',
    'fr_diazo',
    'fr_nitro_arom',
    'fr_phos_ester'
]

# 화학적 특성 디스크립터 계산 함수
def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    #문자열을 RDKit의 분자 구조 객체(Mol)로 변환
    if mol is None:
        return [None] * len(desc_names)
        #유효하지 않은 SMILES일 경우, None으로 채운 리스트를 반환
    return [desc[1](mol) for desc in Descriptors.descList if desc[0] not in useless_cols]
    #Descriptions.descList = RDKit이 제공하는 약 200개 이상의 화학 디스크립터가 포함된 리스트
    #화학 디스크립터중 첫번째가 쓸모없는 거에 포함되지 않았다면 디스크립터 두번째에 mol를 넣어서 실행함
    #디스크립터의 첫번째가  디스크립터 이름, 두번째가 디스크립터 함수 인가볾

# 그래프 기반 특성 추출 함수
def compute_graph_features(smiles, graph_feats):
    mol = Chem.MolFromSmiles(smiles)
    #Mol 객체로 변환
    adj = rdmolops.GetAdjacencyMatrix(mol)
    #분자의 원자들 간 결합 관계를 나타내는 인접 행렬 생성
    G = nx.from_numpy_array(adj)
    #인접 행렬을 기반으로 NetworkX 그래프 생성
    #원자 → 노드, 결합 → 엣지

    graph_feats['graph_diameter'].append(nx.diameter(G) if nx.is_connected(G) else 0)
    #그래프 지름 (graph diameter)
    #가장 먼 두 노드 사이의 최단거리
    #그래프가 연결되어 있어야(diameter 정의 가능) 계산하고, 그렇지 않으면 0
    graph_feats['avg_shortest_path'].append(nx.average_shortest_path_length(G) if nx.is_connected(G) else 0)
    #평균 최단 경로 길이 (average shortest path)
    #그래프의 모든 노드 쌍 사이의 평균 최단 경로 길이
    #역시 연결된 그래프일 경우에만 계산
    graph_feats['num_cycles'].append(len(list(nx.cycle_basis(G))))
    #사이클 개수 (number of cycles)
    #사이클 (ring 구조)의 개수 계산
    #nx.cycle_basis()는 그래프의 모든 독립 사이클을 반환 → 그 길이를 계산하여 개수 측정

# 전처리 전체 적용 함수
def preprocessing(df):
    global desc_names
    desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]
    #molecular descriptor 중 useless_cols에 해당하지 않는 디스크립터 이름만 추림
    descriptors = [compute_all_descriptors(smi) for smi in df['SMILES'].to_list()]
    #각 분자의 디스크립터 리스트를 모아 descriptors 생성
    #2차원 리스트

    graph_feats = {'graph_diameter': [], 'avg_shortest_path': [], 'num_cycles': []}
    #그래프 특성 값을 담을 딕셔너리 초기화
    #(각 항목은 리스트 형태이며 분자 수만큼 값이 들어감)
    for smile in df['SMILES']:
        compute_graph_features(smile, graph_feats)
        #각 SMILES에 대해 그래프 특성 추출 함수 적용하여, graph_feats 딕셔너리에 값 채워 넣음

    result = pd.concat([
        pd.DataFrame(descriptors, columns=desc_names),
        pd.DataFrame(graph_feats)
    ], axis=1)
    #디스크립터 + 그래프 특성을 데이터프레임으로 변환 후 가로 방향으로 합침
    result = result.replace([-np.inf, np.inf], np.nan)
    #무한대 값이 존재할 경우 결측치로 처리
    return result

# 전처리 수행 후 병합
train = pd.concat([train, preprocessing(train)], axis=1)
test = pd.concat([test, preprocessing(test)], axis=1)

# 각 타겟에 대해 불변 컬럼 제거
all_features = train.columns[7:].tolist()
#train 데이터의 앞쪽 7개 열을 제외한 나머지 열들을 후보 feature 목록으로 가져옮
features = {}
#타깃 변수별로 사용할 feature 목록을 저장할 딕셔너리
for target in CFG.TARGETS:
    const_descs = []
    #제거 대상을 담아둘 리스트
    for col in train.columns.drop(CFG.TARGETS):
        #타깃 컬럼들은 제외하고 나머지 모든 열을 대상으로 순회
        if train[train[target].notnull()][col].nunique() == 1:
            #col(feature)의 값이 모두 하나의 값만 가지는 경우 = 상수 feature
            const_descs.append(col)
            #제거대상에 포함
    features[target] = [f for f in all_features if f not in const_descs]
    #상수 feature들을 제외한 나머지를 해당 타깃의 학습 feature로 설정

# MAE 정의
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# LGBM 기본 파라미터 설정
base_params = {
    'device_type': 'cpu',
    'n_estimators': 1_000_000,
    'objective': 'regression_l1',
    'metric': 'mae',
    'verbosity': -1,

    'num_leaves': 50,
    'min_data_in_leaf': 2,
    'learning_rate': 0.01,
    'max_bin': 500,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'lambda_l1': 2,
    'lambda_l2': 2,
}

# 각 타겟별로 모델 학습 및 예측
for target in CFG.TARGETS:
    print(f'\n\nTARGET {target}')
    train_part = train[train[target].notnull()].reset_index(drop=True)
    train[f'{target}_pred'] = 0
    test[target] = 0
    oof_lgb = np.zeros(len(train_part))
    scores = []

    kf = KFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    for i, (trn_idx, val_idx) in enumerate(kf.split(train_part, train_part[target])):
        print(f"\n--- Fold {i + 1} ---")

        x_trn = train_part.loc[trn_idx, features[target]]
        y_trn = train_part.loc[trn_idx, target]
        x_val = train_part.loc[val_idx, features[target]]
        y_val = train_part.loc[val_idx, target]

        model_lgb = lgb.LGBMRegressor(**base_params)
        model_lgb.fit(
            x_trn, y_trn,
            eval_set=[(x_val, y_val)],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=300,
                    verbose=False,
                ),
                lgb.log_evaluation(2500)
            ],
        )

        with open(f'/kaggle/working/lgb_{target}_fold_{i}.pkl', 'wb') as f:
            pickle.dump(model_lgb, f)

        val_preds = model_lgb.predict(x_val, num_iteration=model_lgb.best_iteration_)
        score = mae(y_val, val_preds)
        scores.append(score)
        print(f'MAE: {np.round(score, 5)}')

        oof_lgb[val_idx] = val_preds
        test[target] += model_lgb.predict(
            test[features[target]],
            num_iteration=model_lgb.best_iteration_
        ) / CFG.FOLDS

    train.loc[train[target].notnull(), f'{target}_pred'] = oof_lgb

    print(f'\nMean MAE: {np.round(np.mean(scores), 5)}')
    print(f'Std MAE: {np.round(np.std(scores), 5)}')
    print('-' * 30)

# 예측값과 실제값 비교 시각화
for t in CFG.TARGETS:
    preds = train[train[t].notnull()][f'{t}_pred']
    vals = train[train[t].notnull()][t]
    line_min = min(preds.min(), vals.min())
    line_max = max(preds.max(), vals.max())

    sns.scatterplot(x=preds, y=vals, alpha=0.5)
    plt.plot(
        [line_min, line_max],
        [line_min, line_max],
        color='red',
        linewidth=2,
        linestyle='dashed'
    )
    plt.show()

# 평가 메트릭 (wMAE) 정의 및 적용
MINMAX_DICT =  {
        'Tg': [-148.0297376, 472.25],
        'FFV': [0.2269924, 0.77709707],
        'Tc': [0.0465, 0.524],
        'Density': [0.748691234, 1.840998909],
        'Rg': [9.7283551, 34.672905605],
    }
NULL_FOR_SUBMISSION = -9999

def scaling_error(labels, preds, property):
    error = np.abs(labels - preds)
    min_val, max_val = MINMAX_DICT[property]
    label_range = max_val - min_val
    return np.mean(error / label_range)

def get_property_weights(labels):
    property_weight = []
    for property in MINMAX_DICT.keys():
        valid_num = np.sum(labels[property] != NULL_FOR_SUBMISSION)
        property_weight.append(valid_num)
    property_weight = np.array(property_weight)
    property_weight = np.sqrt(1 / property_weight)
    return (property_weight / np.sum(property_weight)) * len(property_weight)

def wmae_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    chemical_properties = list(MINMAX_DICT.keys())
    property_maes = []
    property_weights = get_property_weights(solution[chemical_properties])
    for property in chemical_properties:
        is_labeled = solution[property] != NULL_FOR_SUBMISSION
        property_maes.append(scaling_error(solution.loc[is_labeled, property], submission.loc[is_labeled, property], property))

    if len(property_maes) == 0:
        raise RuntimeError('No labels')
    return float(np.average(property_maes, weights=property_weights))

tr_solution = train[['id'] + CFG.TARGETS]
tr_submission = train[['id'] + [t + '_pred' for t in CFG.TARGETS]]
tr_submission.columns = ['id'] + CFG.TARGETS
print(f"wMAE: {round(wmae_score(tr_solution, tr_submission, row_id_column_name='id'), 5)}")


# SMILES 중복 제거로 test 데이터의 known 값 채움
for t in CFG.TARGETS:
    for s in train[train[t].notnull()]['SMILES']:
        if s in test['SMILES'].tolist():
            test.loc[test['SMILES']==s, t] = train[train['SMILES']==s][t].values[0]

# 최종 제출 파일 저장
test[['id'] + CFG.TARGETS].to_csv('submission.csv', index=False)
