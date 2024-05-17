import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Load the dataset
data = pd.read_csv('final2.csv')

# Create a label-to-index mapping
label_map = {label: i for i, label in enumerate(data['Symptoms'].unique())}

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))

# Tokenize the input data
input_ids = []
attention_masks = []
labels = []

for statement, symptoms in zip(data['Statement'], data['Symptom']):
    encoded_dict = tokenizer.encode_plus(
        statement,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    labels.append(label_map[symptoms])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Create the data loader
dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

batch_size = 16
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Set up the optimizer and training loop
optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)

        model.zero_grad()

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_masks,
            labels=batch_labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    average_loss = total_loss / len(train_dataloader)

    # Evaluation
    model.eval()
    val_loss = 0
    num_correct = 0
    num_total = 0

    for batch in val_dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                labels=batch_labels
            )

        val_loss += outputs.loss.item()

        _, predicted_labels = torch.max(outputs.logits, dim=1)
        num_correct += torch.sum(predicted_labels == batch_labels).item()
        num_total += batch_labels.size(0)

    accuracy = num_correct / num_total
    average_val_loss = val_loss / len(val_dataloader)

    print(f'Epoch: {epoch+1}/{epochs}')
    print(f'Training loss: {average_loss:.4f}')
    print(f'Validation loss: {average_val_loss:.4f}')
    print(f'Validation accuracy: {accuracy:.4f}\n')

# Save the fine-tuned model
model.save_pretrained('fine_tuned_model')
