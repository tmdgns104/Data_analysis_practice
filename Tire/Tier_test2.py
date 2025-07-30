from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


# NR, SBR, BR(타이어용 고분자)의 SMILES 분자 구조 → RDKit을 이용해 ECFP (Morgan fingerprint) 벡터로 변환
# 각각의 조성비(NR/SBR/BR)를 반영해 ECFP 벡터의 가중 평균 벡터 생성
# 이렇게 만든 벡터를 X로, 유리전이온도 Tg를 y로 해서 학습 데이터 생성
# RandomForestRegressor로 모델을 학습하고, 예측 정확도 평가 (R², RMSE 출력)
# 예측값과 실제값의 산점도 그래프 시각화



# 1 각 고분자의 SMILES를 RDKit으로 분자 객체로 변환	                "C=CC(C)C" → RDKit Mol 객체
# 2	GetMorganFingerprintAsBitVect 사용하여 2048차원 벡터 생성	    2048길이의 0과 1로 이루어진 배열
# 3	NR, SBR, BR 각각의 벡터에 조성비를 곱해서 평균 벡터 생성	        NR: 0.4, SBR: 0.3, BR: 0.3
# 4	평균 벡터를 X, 문헌 기반 Tg 계산 결과를 y로 사용	                ML 데이터셋 완성
# 5	X, y를 가지고 랜덤포레스트 모델 학습	                            분자 구조 기반 Tg 예측
# 6	예측값 vs 실제 Tg를 비교해서 정확도 평가	                        R², RMSE, 산점도 시각화







#Morgan Fingerprint
#원자 중심의 원형 서브구조를 기반으로 분자의 특징을 추출
#반지름(radius)을 기준으로 중심 원자 주변의 원자들을 포함한 서브구조를 생성
#각 서브구조는 해시되어 특정 비트 위치에 매핑
# nBits는 생성할 벡터의 길이
# radius는 서브구조를 생성할때 고려할 반지름




# RDKit로 SMILES → ECFP 벡터
def get_ecfp(smiles, radius=2, n_bits=2048):
    # 몰 객체로 변환
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
        #Smiles로 해석을 못하는건 예외처리
    # 분자의 Morgan 지문(Morgan fingerprint)을 비트 벡터(bit vector) 형태로 생성
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

# 1. 고분자 단위의 SMILES
smiles_dict = {
    "NR": "C=CC(C)C",      # Isoprene 기반
    "SBR": "c1ccccc1",     # Styrene
    "BR": "C=CC=C"         # Butadiene
}

# 2. 무작위 조성 생성 (NR+SBR+BR=1)
def generate_compositions(n_samples=2000):
    compositions = []
    for _ in range(n_samples):
        nr = np.random.uniform(0.0, 1.0)
        sbr = np.random.uniform(0.0, 1.0 - nr)
        br = 1.0 - nr - sbr
        compositions.append((round(nr, 3), round(sbr, 3), round(br, 3)))
    return compositions

# 3. Tg 생성 함수 (문헌 기반 가중 평균 + 노이즈)
def estimate_tg(nr, sbr, br):
    tg = nr * 243 + sbr * 234 + br * 210 + np.random.normal(0, 1)
    return round(tg, 2)

# 4. 학습용 데이터 생성
X = []
y = []

compositions = generate_compositions(2000)
ecfp_cache = {k: get_ecfp(v) for k, v in smiles_dict.items()}

for nr, sbr, br in compositions:
    weighted_fp = (
        ecfp_cache["NR"] * nr +
        ecfp_cache["SBR"] * sbr +
        ecfp_cache["BR"] * br
    )
    X.append(weighted_fp)
    y.append(estimate_tg(nr, sbr, br))

X = np.array(X)
y = np.array(y)

# 5. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. RandomForest 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. 예측
y_pred = model.predict(X_test)

# 8. 평가
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"📈 R²: {r2:.4f}")
print(f"📉 RMSE: {rmse:.4f}")

# 9. 예측 vs 실제 산점도
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("Actual Tg (K)")
plt.ylabel("Predicted Tg (K)")
plt.title(f"Random Forest Prediction\nR² = {r2:.3f}, RMSE = {rmse:.2f}")
plt.grid(True)
plt.tight_layout()
plt.show()
