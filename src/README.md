# Shared code

Code that more than one experiment uses:

| File | What it does |
| --- | --- |
| `preprocessing/tokenizing.py` | Loads and tokenizes IMDB, Amazon and Finance |
| `preprocessing/create_ds.py` | The vocabulary, and the same-dataset pairs of `experiments/documents` |
| `common.py` | The train and test splits, set-of-words matrices and the pretext pairs |
| `family_readout.py` | Reads OR groups out as word families, and turns families into input bits |

The experiment scripts add the project root to `sys.path`, so nothing needs to be installed.
