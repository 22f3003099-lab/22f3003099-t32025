import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from models.bert_classifier import BertForMultiLabel
from transformers import AutoTokenizer
from utils.dataset import EmotionDataset
from torch.utils.data import DataLoader

emotion_cols = ['anger','fear','joy','sadness','surprise']

def run_inference(model_path, test_csv, out_csv="submission.csv"):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_df = pd.read_csv(test_csv)
    test_ds = EmotionDataset(test_df['text'].tolist(), None, tokenizer, max_len=128)
    test_loader = DataLoader(test_ds, batch_size=16)

    model = BertForMultiLabel(out_dim=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            logits = model(batch['input_ids'], batch['attention_mask'])
            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
            preds_list.append(preds)

    preds = np.vstack(preds_list)

    sub = pd.DataFrame(preds, columns=emotion_cols)
    sub.insert(0, "id", test_df['id'])
    sub.to_csv(out_csv, index=False)

    print("Saved:", out_csv)
