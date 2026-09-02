import csv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

texts = []
labels = []

# Read CSV file
with open("fake_news_dataset_1200.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        texts.append(row["text"])
        labels.append(row["label"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.3,
    random_state=42
)

# Convert text into numerical values
vectorizer = TfidfVectorizer(stop_words="english")

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vectors, y_train)

# Test model
predictions = model.predict(X_test_vectors)
print("Accuracy:", accuracy_score(y_test, predictions) * 100)

# Save the trained model and vectorizer so the Streamlit app can load them
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Saved model.pkl and vectorizer.pkl")
