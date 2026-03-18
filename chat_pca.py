import spacy
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# ----------------------------
# LOAD TEXT DATASET
# ----------------------------

text = open("book.txt", encoding="utf8").read()

doc = nlp(text)

# Extract sentences
sentences = [sent.text.strip() for sent in doc.sents]

# Filter sentences that are too short or too long
sentences = [s for s in sentences if 5 < len(s.split()) < 40]

# Randomize and select 300 sentences
random.shuffle(sentences)
sentences = sentences[:300]

print("Total sentences used:", len(sentences))

# ----------------------------
# ANALYZE SENTENCES
# ----------------------------

data = []

for sent in sentences:

    doc = nlp(sent)

    length = len(doc)

    nouns = sum(1 for token in doc if token.pos_ == "NOUN")
    verbs = sum(1 for token in doc if token.pos_ == "VERB")
    adjectives = sum(1 for token in doc if token.pos_ == "ADJ")
    adverbs = sum(1 for token in doc if token.pos_ == "ADV")

    # Dependency depth (syntactic hierarchy)
    depth = max([len(list(token.ancestors)) for token in doc])

    clauses = verbs

    data.append({
        "sentence": sent,
        "length": length,
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjectives,
        "adverbs": adverbs,
        "clauses": clauses,
        "dependency_depth": depth
    })

df = pd.DataFrame(data)

print("\nSentence Data Sample:")
print(df.head())

# ----------------------------
# LENGTH CATEGORY
# ----------------------------

def length_category(length):
    if length <= 8:
        return "short"
    elif length <= 16:
        return "medium"
    else:
        return "long"

df["length_category"] = df["length"].apply(length_category)

# ----------------------------
# GROUP STATISTICS
# ----------------------------

group_stats = df.groupby("length_category").mean(numeric_only=True)

print("\nAverage Complexity by Sentence Length:")
print(group_stats)

# ----------------------------
# CORRELATION ANALYSIS
# ----------------------------

correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)

# Save dataset
df.to_csv("sentence_analysis.csv", index=False)

# ----------------------------
# SCATTER GRAPH
# ----------------------------

plt.figure()

plt.scatter(df["length"], df["dependency_depth"])

plt.title("Sentence Length vs Syntactic Complexity")
plt.xlabel("Sentence Length (words)")
plt.ylabel("Dependency Depth")

plt.show()

# ----------------------------
# REGRESSION ANALYSIS
# ----------------------------

X = df["length"].values.reshape(-1,1)
y = df["dependency_depth"].values

model = LinearRegression()
model.fit(X,y)

y_pred = model.predict(X)

plt.figure()

plt.scatter(df["length"], df["dependency_depth"])
plt.plot(df["length"], y_pred)

plt.title("Regression: Sentence Length vs Syntactic Complexity")
plt.xlabel("Sentence Length")
plt.ylabel("Dependency Depth")

plt.show()

print("Regression slope:", model.coef_[0])
print("Regression intercept:", model.intercept_)