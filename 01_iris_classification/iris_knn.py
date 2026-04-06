import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv('Iris.csv')
print('First 5 rows of dataset:')
print(df.head())

# Clean the corrupted file
df = df[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']]
df = df.dropna(subset=['Species'])

# Features and target
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))
