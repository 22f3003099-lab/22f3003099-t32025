import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from utils.dataset import EmotionDataset

emotion_cols = ['anger','fear','joy','sadness','surprise']

def prepare_data(train_csv, test_csv, max_len=128, batch_size=16):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    train['text'] = train['text'].astype(str)
    test['text'] = test['text'].astype(str)

    train['label_sum'] = train[emotion_cols].sum(axis=1)
    train_df, val_df = train_test_split(train, test_size=0.15, stratify=train['label_sum'], random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_ds = EmotionDataset(train_df['text'].tolist(), train_df[emotion_cols].values, tokenizer, max_len)
    val_ds   = EmotionDataset(val_df['text'].tolist(), val_df[emotion_cols].values, tokenizer, max_len)
    test_ds  = EmotionDataset(test['text'].tolist(), None, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader
