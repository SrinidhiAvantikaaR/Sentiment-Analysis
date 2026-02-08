from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import joblib

#Import CSV
df = pd.read_csv("D:\Project1\Sentiment-Analysis\data\cleaned_sm.csv")
#Remove null values
df = df.dropna(subset=['text']).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

#Vectorization 
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range= (1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

#validate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))


# Save model and vectorizer
joblib.dump(model, 'social_media_model.joblib')
joblib.dump(vectorizer, 'social_media_vectorizer.joblib')