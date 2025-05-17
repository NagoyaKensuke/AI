import re
from collections import Counter

# サンプルのレビューとその感情（1: ポジティブ, 0: ネガティブ）
reviews = [
    "I love this product!",
    "This is awful, I hate it",
    "Fantastic, would buy again",
    "Not great, better options available",
    "Five stars, highly recommend",
    "One star, very disappointing",
]

labels = [1, 0, 1, 0, 1, 0]


def tokenize(text: str):
    """簡易的なトークナイザー"""
    return re.findall(r"\b\w+\b", text.lower())


def train(reviews, labels):
    """単純なナイーブベイズ分類器の訓練"""
    pos_counts = Counter()
    neg_counts = Counter()
    pos_total = neg_total = 0

    for review, label in zip(reviews, labels):
        words = tokenize(review)
        if label == 1:
            pos_counts.update(words)
            pos_total += len(words)
        else:
            neg_counts.update(words)
            neg_total += len(words)

    vocab = set(pos_counts) | set(neg_counts)
    return pos_counts, neg_counts, pos_total, neg_total, vocab


def predict(review, pos_counts, neg_counts, pos_total, neg_total, vocab):
    """ナイーブベイズ分類器による予測"""
    words = tokenize(review)
    pos_prob = neg_prob = 1.0
    v_size = len(vocab)

    for word in words:
        pos_prob *= (pos_counts[word] + 1) / (pos_total + v_size)
        neg_prob *= (neg_counts[word] + 1) / (neg_total + v_size)

    return 1 if pos_prob >= neg_prob else 0


# モデルの訓練
pos_counts, neg_counts, pos_total, neg_total, vocab = train(reviews, labels)

# 新しいレビューでテスト
new_reviews = [
    "This is great!",
    "I hate this",
    "Not recommended",
    "Two thumbs up!",
    "Quite disappointing",
]

predicted = [
    predict(r, pos_counts, neg_counts, pos_total, neg_total, vocab) for r in new_reviews
]

# 結果の表示
for review, label in zip(new_reviews, predicted):
    sentiment = "ポジティブ" if label == 1 else "ネガティブ"
    print(f"レビュー: '{review}' → 感情: {sentiment}")
