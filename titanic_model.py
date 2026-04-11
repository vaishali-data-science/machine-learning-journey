import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df = pd.read_csv('train.csv')

# Select columns
df = df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

# Fill missing values (correct way)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert text to numbers
df['Sex'] = df['Sex'].map({'male':0,'female':1})
df['Embarked'] = df['Embarked'].map({'S':0,'C':1,'Q':2})

print(df.isnull().sum())  # MUST be all 0

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))
print('\nClassification Report:\n')
print(classification_report(y_test,y_pred))