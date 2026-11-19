# Data

The datasets are not part of the project. Download them and place them here:

| File | Dataset |
| --- | --- |
| `raw/IMDB-Dataset.csv` | IMDB Dataset of 50K Movie Reviews, Kaggle (Lakshmipathi N.) |
| `raw/amazon-reviews/train.csv`, `raw/amazon-reviews/test.csv` | Amazon Reviews, Kaggle (K. Jain) |
| `raw/finance_news.csv` | Sentiment Analysis for Financial News, Kaggle (A. Sinha) |

`processed/splits_seed42.pkl` is created on the first run. It holds the tokenized train and test
splits, and rebuilding it reads the 3.6-million-row Amazon file once.
