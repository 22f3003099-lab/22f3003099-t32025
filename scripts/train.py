import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np

from models.bert_classifier import BertForMultiLabel
from models.lstm_classifier import LSTMClassifier
from models.schedulers import get_scheduler

emotion_cols = ['anger','fear','joy','sadness','surprise']

def train(model, train_loader, val_loader, lr=2e-5, epochs=3, scheduler_type=None):
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    scheduler = None
    if scheduler_type:
        scheduler = get_scheduler(optimizer, scheduler_type)

    best_macro = 0
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            labels = batch['labels']

            if 'input_ids' in batch:
                logits = model(batch['input_ids'], batch['attention_mask'])
            else:
                logits = model(batch['text'])

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

        macro_f1 = evaluate(model, val_loader)
        print(f"Epoch {epoch+1} | Macro F1 = {macro_f1:.4f}")

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            labels = batch['labels'].numpy()

            logits = model(batch['input_ids'], batch['attention_mask'])
            preds = torch.sigmoid(logits).numpy()

            y_true.append(labels)
            y_pred.append((preds >= 0.5).astype(int))

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    return np.mean([f1_score(y_true[:,i], y_pred[:,i]) for i in range(len(emotion_cols))])
