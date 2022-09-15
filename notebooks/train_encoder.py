import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from vit.encoder import TransformerEncoderClassifer

from transformers import AutoTokenizer
import datasets 
from tqdm import tqdm # tqdm is a python progress bar library

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Training configurations
num_epoch = 100
learning_rate = 1e-5
optimizer = 'adam'
device='cuda:0'

# Data configurations
batch_size=150

# Model configurations
vocab_size=tokenizer.vocab_size
d_model=256
num_layer=2
num_head=8
d_k=32
dropout_rate=0.1
num_class=2

imdb = datasets.load_dataset('imdb')
# imdb

# Tokenization with the `batch_encode_plus` function
imdb['train'] = imdb['train'].map(
    lambda x: tokenizer.batch_encode_plus(
        x['text'], padding=False, return_attention_mask=False, truncation=True), 
    batched=True)
imdb['test'] = imdb['test'].map(
    lambda x: tokenizer.batch_encode_plus(
        x['text'], padding=False, return_attention_mask=False, truncation=True), 
    batched=True)

def collate_fn(batch):
    batch = tokenizer.pad(batch)
    return {
        'input_ids': torch.tensor(batch['input_ids']),
        'attn_mask': torch.tensor(batch['attention_mask']),
        'labels': torch.tensor(batch['label'])
    }

train_dataloader = DataLoader(
    imdb['train'],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers = 4
)

test_dataloader = DataLoader(
    imdb['test'],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers = 4
)

classifier = TransformerEncoderClassifer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layer=num_layer,
    num_head=num_head,
    d_k=d_k,
    dropout_rate=dropout_rate,
    num_class=num_class
).to(device)

optimizer = Adam(classifier.parameters(), lr=learning_rate)


for i in range(num_epoch):
    # Train
    with tqdm(train_dataloader) as train_epoch:
        for batch_id, batch in enumerate(train_epoch):
            input_ids, attn_mask, labels = batch['input_ids'].to(device), batch['attn_mask'].to(device), batch['labels'].to(device)
            # attn_mask = batch['attn_mask'].to(device)
            # labels = batch['labels'].to(device)
            # if batch_id > 3:
            #     break

            outputs = classifier(
                x=input_ids,
                attn_mask=attn_mask
            )
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)

            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == labels).sum().item()
            accuracy = correct / batch_size

            loss.backward()
            optimizer.step()

            # print(loss)
            # print(accuracy)
            train_epoch.set_description(f"Training Epoch {i}")
            train_epoch.set_postfix({
                'Loss': loss.item(), 
                'Accuracy': accuracy
            })
    # Validate
    with tqdm(test_dataloader) as test_epoch:
        for batch_id, batch in enumerate(test_epoch):
            # if batch_id > 3:
            #     break
            input_ids, attn_mask, labels = batch['input_ids'].to(device), batch['attn_mask'].to(device), batch['labels'].to(device)

            # attn_mask = batch['attn_mask'].to(device)
            # labels = batch['labels'].to(device)

            outputs = classifier(
                x=input_ids,
                attn_mask=attn_mask
            )
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)

            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == labels).sum().item()
            accuracy = correct / batch_size

            test_epoch.set_description(f"Test Epoch {i}")
            test_epoch.set_postfix({
                'Loss': loss.item(), 
                'Accuracy': accuracy
            })

