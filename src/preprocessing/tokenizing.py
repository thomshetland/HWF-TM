"""Tokenize the raw reviews and save them to data/processed.

Loaders return rows as {"review": ..., "sentiment": ...} so tokenize_document works on any dataset.
"""

import csv
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW = DATA_DIR / "raw" / "IMDB-Dataset.csv"
FINANCE_RAW = DATA_DIR / "raw" / "finance_news.csv"
AMAZON_RAW = DATA_DIR / "raw" / "amazon-reviews" / "train.csv"
PROCESSED = DATA_DIR / "processed" / "IMDB-Dataset-tokenized.csv"

# Reviews contain HTML line breaks ("<br /><br />") that would otherwise become tokens.
HTML_TAG = re.compile(r"<[^>]+>")
# Words, keeping inner apostrophes so "don't" stays one token.
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")


def tokenize(text):
    text = HTML_TAG.sub(" ", text).lower()
    return TOKEN.findall(text)


def tokenize_document(document, save_document=True):
    tokenized_document = []
    for row in document:
        tokenized_document.append(
            {"tokens": " ".join(tokenize(row["review"])), "sentiment": row["sentiment"]}
        )

    if save_document:
        PROCESSED.parent.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["tokens", "sentiment"])
            writer.writeheader()
            writer.writerows(tokenized_document)

    return tokenized_document


def load_imdb(path=RAW):
    csv.field_size_limit(sys.maxsize)  # some reviews exceed the default 131k field limit
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_finance(path=FINANCE_RAW):
    # No header, (sentiment, sentence) columns, latin-1 encoded, CR line endings.
    with open(path, newline="", encoding="latin-1") as f:
        return [{"review": text, "sentiment": sentiment} for sentiment, text in csv.reader(f)]


def load_amazon(n, rng, path=AMAZON_RAW):
    """A uniform random sample of n reviews, read in one pass.

    The file has 3.6M rows, so reservoir sampling keeps memory at n rows. Columns are
    (polarity, title, text) with no header; polarity 1 = negative, 2 = positive. Only the
    body text is kept, because IMDB reviews have no title.
    """
    sample = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, (polarity, _title, text) in enumerate(csv.reader(f)):
            row = {"review": text, "sentiment": "positive" if polarity == "2" else "negative"}
            if i < n:
                sample.append(row)
            else:
                j = rng.integers(i + 1)
                if j < n:
                    sample[j] = row
    return sample


if __name__ == "__main__":
    tokenized = tokenize_document(load_imdb(), save_document=True)
    print(f"Tokenized {len(tokenized)} reviews -> {PROCESSED}")
