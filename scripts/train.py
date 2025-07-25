import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
import argparse
import os
import pandas as pd

class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        description = str(self.data.Description.iloc[index])
        description = " ".join(description.split())
        inputs = self.tokenizer.encode_plus(
            description,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        
        target = torch.tensor(self.data.Target.iloc[index], dtype=torch.float)
        return {'ids': ids, 'mask': mask, 'targets': target}

    def __len__(self):
        return len(self.data)

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output.last_hidden_state
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        return self.classifier(pooler)

def train_epoch(model, loader, optimizer, loss_fn, device, scaler, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device).unsqueeze(1)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(ids, mask)
            loss = loss_fn(outputs, targets)
        
        if torch.isnan(loss):
            print("Pérdida es NaN. Saltando este paso de entrenamiento.")
            continue

        if torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    print(f"Training Loss: {avg_loss:.4f}")

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['targets'].to(device).unsqueeze(1)
            outputs = model(ids, mask)
            loss = loss_fn(outputs, targets)
            
            if torch.isnan(loss):
                print("Pérdida de validación es NaN. Saltando este batch.")
                continue

            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds.float() == targets).sum().item()

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    accuracy = correct / len(loader.dataset) if len(loader.dataset) > 0 else 0
    print(f"Validation Loss: {avg_loss:.4f} | Validation Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train-batch-size', type=int, default=16)
    parser.add_argument('--valid-batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--max-len', type=int, default=128)
    parser.add_argument('--warmup-steps', type=int, default=0)
    
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(os.path.join(args.data_dir, 'credir_risk_reto_classified.csv'))
    if df['Target'].dtype == 'object':
        df['Target'] = df['Target'].map({'good risk': 1.0, 'bad risk': 0.0})

    train_df = df.sample(frac=0.8, random_state=200).reset_index(drop=True)
    val_df = df.drop(train_df.index).reset_index(drop=True)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_ds = NewsDataset(train_df, tokenizer, args.max_len)
    val_ds = NewsDataset(val_df, tokenizer, args.max_len)

    num_workers = 2 

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.valid_batch_size, shuffle=False, num_workers=num_workers)

    model = DistilBERTClass().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    
    num_training_steps = len(train_loader) * args.epochs
    actual_warmup_steps = min(args.warmup_steps, num_training_steps)
    if actual_warmup_steps != args.warmup_steps:
        print(f"ADVERTENCIA: warmup_steps ({args.warmup_steps}) recortado a {actual_warmup_steps} para no exceder los pasos totales ({num_training_steps})")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=actual_warmup_steps,
        num_training_steps=num_training_steps
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.50 

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_epoch(model, train_loader, optimizer, loss_fn, device, scaler, scheduler)
        val_acc = eval_epoch(model, val_loader, loss_fn, device)
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"Nueva mejor accuracy: {best_acc:.4f}. Guardando modelo...")
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model_best.bin'))
            tokenizer.save_pretrained(args.model_dir)

    print("\nTraining finished.")

if __name__ == '__main__':
    main()