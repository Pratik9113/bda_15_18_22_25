import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. Generate Stock Data
# =========================
np.random.seed(42)
data_size = 300

price = np.cumsum(np.random.randn(data_size)) + 100

df = pd.DataFrame({'price': price})

# =========================
# 2. Feature Engineering
# =========================
df['returns'] = df['price'].pct_change()

df['ma_5'] = df['price'].rolling(5).mean()
df['ma_10'] = df['price'].rolling(10).mean()

df['ma_diff'] = df['ma_5'] - df['ma_10']   # trend strength

df['volatility'] = df['returns'].rolling(5).std()

# Drop missing values
df = df.dropna()

# =========================
# 3. Target Creation
# =========================
# 1 = Buy (price goes up next step)
# 0 = Not Buy
df['target'] = (df['price'].shift(-1) > df['price']).astype(int)

df = df.dropna()

# =========================
# 4. Features & Labels
# =========================
features = ['returns', 'ma_diff', 'volatility']
X = df[features]
y = df['target']

# =========================
# 5. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =========================
# 6. Scaling (IMPORTANT)
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 7. Train SVM Model
# =========================
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# =========================
# 8. Predictions
# =========================
y_pred = model.predict(X_test)

# =========================
# 9. Evaluation
# =========================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# 10. Real-Time Prediction
# =========================
latest_data = X.iloc[-1].values.reshape(1, -1)
latest_data_scaled = scaler.transform(latest_data)

prediction = model.predict(latest_data_scaled)

if prediction[0] == 1:
    print("\n📈 BUY SIGNAL")
else:
    print("\n📉 DO NOT BUY")
