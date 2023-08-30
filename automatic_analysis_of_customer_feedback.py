from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# サンプルのレビューとその感情（1: ポジティブ, 0: ネガティブ）
reviews = ["I love this product!", 
           "This is awful, I hate it", 
           "Fantastic, would buy again", 
           "Not great, better options available", 
           "Five stars, highly recommend", 
           "One star, very disappointing"]

labels = [1, 0, 1, 0, 1, 0]

# モデルの訓練
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(reviews, labels)

# 新しいレビューでテスト
new_reviews = ["This is great!", "I hate this", "Not recommended", "Two thumbs up!", "Quite disappointing"]
predicted = model.predict(new_reviews)

# 結果の表示
for review, label in zip(new_reviews, predicted):
    sentiment = "ポジティブ" if label == 1 else "ネガティブ"
    print(f"レビュー: '{review}' → 感情: {sentiment}")
