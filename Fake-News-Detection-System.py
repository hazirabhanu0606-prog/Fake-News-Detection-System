import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("news.csv")

# Split data
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Check accuracy
y_pred = model.predict(X_test_vec)
print(f"\n📊 Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n")

# 🔁 Loop for user input
while True:
    news = input("\n📝 Enter a news headline (or type 'exit' to quit): ")

    if news.lower() == "exit":
        print("👋 Exiting program...")
        break

    news_vec = vectorizer.transform([news])
    prediction = model.predict(news_vec)

    print("\n🔍 Checking News...\n")

    if prediction[0] == 1:
        print("✅ Result: This news is REAL\n")
    else:
        print("❌ Result: This news is FAKE\n")