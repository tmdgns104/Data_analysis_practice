import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 약어	이름	                 전체 이름	                      종류	        주요 역할
# NR	천연고무	             Natural Rubber	                  천연	        탄성, 강도, 복원력
# SBR	스티렌-부타디엔 고무     Styrene-Butadiene Rubber	      합성	        내마모성, 젖은 노면 접지력
# BR	부타디엔 고무	         Butadiene Rubber	              합성	        저온 유연성, 회전저항 감소


#1. 유리전이온도 (Tg, Glass Transition Temperature)
#   타이어의 탄성, 접지력, 연비에 직접적인 영향
#   Tg가 낮으면 겨울철에서 유연, 높으면 고속 주행 시 성능 유지

#2. 밀도 (Density)
#   무게, 연비, 진동 흡수 성능과 연관
#   타이어 설계에서 구조 강성 및 무게 최적화에 필요

#3. 탄성률,영율 (Elastic Modulus, 또는 Young’s Modulus)
#   타이어의 변형 저항성, 핸들링 특성
#   너무 낮으면 변형 많고, 너무 높으면 승차감 저하

#4. 손실탄젠트 (tan δ at 60°C 또는 0°C)
#   60°C에서의 tan δ: 연비 (낮을수록 좋음)
#   0°C에서의 tan δ: 젖은 노면 접지력 (높을수록 좋음)
#   실제 타이어 제조사들이 가장 중요하게 보는 특성 중 하나


#5. 인장 강도 (Tensile Strength), 신율 (Elongation at Break)
#   타이어가 얼마나 늘어나고 찢어지지 않는지를 나타냄
#   내구성과 밀접한 연관




# 특성명	                        중요도	            	연관 성능
# Tg           (유리전이온도)	    ★★★★☆	            탄성, 고온 성능, 접지력
# Density      (밀도)	        ★★★☆☆		        무게, 연비, 진동
# Elasticity   (탄성률)	        ★★★☆☆	            핸들링, 구조 강성
# tan δ	       (손실탄센트)       ★★★★★	 	        연비, 접지력
# Tensile      (인장강도)	        ★★★☆☆		        내구성, 구조적 안정성



#조성에 따른 유리전이 온도
# 데이터 불러오기(데이터는 실제 측정값이 아닌 문헌상의 정보로 데이터를 만들어 냈습니다. 실제와 다를수 있음)
df = pd.read_csv(r".\Data\nr_sbr_br_8000_samples.csv")  # 다운로드 받은 경로에 맞게 수정

# 기본 확인
# print(df.head())
# print(df.describe())



from sklearn.model_selection import train_test_split

# 입력 변수 (조성비)
X = df[['NR', 'SBR', 'BR']]

# 예측 대상: Tg(유리전이 온도)
y = df['Tg_K']

# 학습/테스트 분리 (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train, X_test, y_train, y_test )




from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#회귀분석 모델 호출
lr_model = LinearRegression()
#학습 , 피팅
lr_model.fit(X_train, y_train)

#예측
y_pred_lr = lr_model.predict(X_test)

#예측값과 실제 값(test에대한) 으로 mse계산(오차제곱평균)
mse_lr = mean_squared_error(y_test, y_pred_lr)
#rmse 계산, 루트 씌우기
rmse_lr = np.sqrt(mse_lr)

# 평가
#R² (결정계수, Coefficient of Determination) 1에 가까울수록 모델의 설명력이 높다 R^2 = 1-(잔차제곱합/전체제곱합) 잔차:실제값 - 예측값
print("📈 선형회귀 R2:", r2_score(y_test, y_pred_lr)) #0.98로 98% 높은 설명률
#rmse
print("📉 RMSE:", rmse_lr)  #1.005 1에 가까우니 오차가 적음






from sklearn.ensemble import RandomForestRegressor
#랜덤포레스트 모델 호출 #(결정 트리(Decision Tree)의 개수,무작위 요소 고정)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

#예측값과 실제 값(test에대한) 으로 mse계산(오차제곱평균)
mse_rf = mean_squared_error(y_test, y_pred_rf)
#rmse 계산, 루트 씌우기
rmse_rf = np.sqrt(mse_rf)

# 평가
print("📈 랜덤포레스트 R2:", r2_score(y_test, y_pred_rf))  #0.97로 97% 높은 설명률
print("📉 RMSE:", rmse_rf)  #






#6인치, 6인치 그래프 생성
plt.figure(figsize=(6, 6))
#산점도 예측 y_test 와 랜덤포레스트 예측값 사이의 산점도 그래프 투명도0.5
plt.scatter(y_test, y_pred_rf, alpha=0.5)
#x=y선 r-- 빨간점선으로
#예측값이 실제값과 완전히 같다면 찍히는 기준선
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test    .max()], 'r--')
plt.xlabel("Actual Tg")
plt.ylabel("Predicted Tg")
plt.title("Random Forest: Actual vs Predicted Tg")
plt.grid(True)
plt.show()








#Feature 중요도 분석
# 모델이 계산한 특성 중요도 배열
importances = rf_model.feature_importances_
feature_names = ['NR', 'SBR', 'BR']

sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (Random Forest)")
plt.show()
#BR이 높은걸 보니(0.5이상),BR 조성이 유리전이온도에 중요한 역확을 함





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 무작위 고분자 조성(NR, SBR, BR)을 생성하는 함수
def generate_compositions(n_samples=8000):
    compositions = []
    for _ in range(n_samples):
        nr = np.random.uniform(0.0, 1.0)                  # NR 비율을 0~1 사이에서 무작위로 선택
        sbr = np.random.uniform(0.0, 1.0 - nr)            # SBR은 전체가 1을 넘지 않게 NR과 합산 제한
        br = 1.0 - nr - sbr                               # BR은 나머지로 계산 → NR+SBR+BR = 1
        compositions.append([round(nr, 3), round(sbr, 3), round(br, 3)])
    return np.array(compositions)


# 2. 조성에 따라 Tg 및 밀도를 계산하는 함수 (문헌 Tg,density 값을 참고하여 데이터 만듦 + 노이즈)
def estimate_properties(nr, sbr, br):
    tg = nr * 243 + sbr * 234 + br * 210 + np.random.normal(0, 1)  # 문헌상의 가중 평균 + 약간의 노이즈
    density = nr * 0.94 + sbr * 0.96 + br * 0.93 + np.random.normal(0, 0.002)
    return round(tg, 2), round(density, 3)


n_samples = 8000
compositions = generate_compositions(n_samples)

data = []
for nr, sbr, br in compositions:
    tg, density = estimate_properties(nr, sbr, br)
    data.append({"NR": nr, "SBR": sbr, "BR": br, "Tg_K": tg, "Density": density})

df_2 = pd.DataFrame(data)

# 2. 시각화: BR 조성에 따른 Tg
plt.figure(figsize=(8, 6))
sns.scatterplot(x="BR", y="Tg_K", data=df_2, alpha=0.3, label="Samples")
sns.regplot(x="BR", y="Tg_K", data=df_2, scatter=False, color='red', label="Trend Line")
plt.title("Tg vs BR Composition")
plt.xlabel("BR Composition")
plt.ylabel("Glass Transition Temperature (K)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()











import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. BR 값을 0.0부터 1.0까지 100개로 나눠서 생성
br_values = np.linspace(0, 1, 100)

# 2. NR, SBR 비율 고정 (예: NR=0.3, SBR=0.3 → BR=0.4 ~ 1.0 사이 가능)
#    BR이 늘어나면 NR+SBR은 줄어들어야 하므로, 여기서는 NR, SBR을 0으로 하고 BR만 0~1로 조절 (단일 영향 확인)
nr_fixed = 0.0
sbr_fixed = 0.0

# 3. 조성 배열 생성
X_br_variation = pd.DataFrame({
    "NR": [nr_fixed] * len(br_values),
    "SBR": [sbr_fixed] * len(br_values),
    "BR": br_values
})

# 4. Tg 예측
y_pred_tg = rf_model.predict(X_br_variation)

# 5. 시각화
plt.figure(figsize=(8, 6))
plt.plot(br_values, y_pred_tg, label="Predicted Tg", color='blue')
plt.xlabel("BR Composition")
plt.ylabel("Predicted Tg (K)")
sns.scatterplot(x=df["BR"], y=df["Tg_K"], alpha=0.3, label="Actual Data")
plt.title("📈 Predicted Tg vs BR Composition (Random Forest)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#유리 전이 온도에 영향을 주는 값을 알아보고 실제로 예측 했을때 비슷하게 그래프를 그려낼수 있을지를 보았다
#다른 특성도 이와 같이 분석해서 알아 내면 될것 같음