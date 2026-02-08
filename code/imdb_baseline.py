from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv("D:\Project1\Sentiment-Analysis\data\cleaned_imdb.csv", encoding='ISO-8859-1')
df = df.dropna(subset=['review']).reset_index(drop=True)
df = df.dropna(subset=['sentiment']).reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range= (1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

#validate
y_pred = model.predict(X_test_vec)
print("Logical Regression: \n" + classification_report(y_test, y_pred))

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# SVM
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
svm_preds = svm_model.predict(X_test_vec)

print("SVM:")
print(classification_report(y_test, svm_preds))
 
import joblib

# Save the vectorizer
joblib.dump(vectorizer, "imdb_vectorizer.joblib")

# Save the model
joblib.dump(model, "imdb_model.joblib")