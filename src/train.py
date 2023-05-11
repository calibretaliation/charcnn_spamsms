import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
from pytorch_lightning.utilities.model_summary import ModelSummary
import sys
import os
from alphabet import *
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())

BATCH_SIZE=32

class CharCNN(pl.LightningModule):
    def __init__(self, input_size, num_classes, alphabet, num_filters=256, kernel_sizes=[7, 7, 3, 3, 3, 3], dropout_prob=0.5):
        super(CharCNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_prob = dropout_prob
        self.alphabet = alphabet
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.input_size, self.num_filters, self.kernel_sizes[0], padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,stride=1),
            nn.Conv1d(self.num_filters, self.num_filters, self.kernel_sizes[1], padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,stride=1)
        ])
        
        for i in range(2, len(self.kernel_sizes)-1):
        # for i in range(1, 4):
            self.conv_layers.extend([
                nn.Conv1d(self.num_filters, self.num_filters, self.kernel_sizes[i], padding=0),
                nn.ReLU(),
                # nn.MaxPool1d(kernel_size=3)
            ])
        self.conv_layers.extend([
                nn.Conv1d(self.num_filters, self.num_filters, self.kernel_sizes[-1], padding=0),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3,stride=1)
            ])
        input_shape = (
            BATCH_SIZE,
            self.input_size,
            len(self.alphabet),
        )
        self.output_dimension = self._get_conv_output(input_shape)
        self.fc_layers = nn.Sequential(
            nn.Linear(self.output_dimension, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, self.num_classes)
        )
    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        output_dimension = x.size(1)
        return output_dimension

    def forward(self, x):
        # print(x.shape)
        # print("len char_to_id: ",len(self.char_to_id))
        # print(self.char_to_id)
        # x = x.transpose(1, 2)
        # print("tranpose")
        # print(x.shape)
        for layer in self.conv_layers:
            x = layer(x)
            # print(layer)
            # print(x.shape)

        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.one_hot_encode(x)
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).sum().item() / len(y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.one_hot_encode(x)
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).sum().item() / len(y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return acc
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    
    def prepare_data(self):
        self.df = pd.read_csv('data/train.csv', names = ["class","trash","text"])
        self.classes = sorted(self.df['class'].unique())
        # print(self.classes)
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        self.id_to_class = {i: c for i, c in enumerate(self.classes)}
        # print(self.class_to_id)
        self.char_to_id = {}
        for text in self.df['text']:
            for c in text:
                if c not in self.char_to_id:
                    self.char_to_id[c] = len(self.char_to_id)
        X = self.df['text']
        y = self.df['class']

        # Split the DataFrame into training and validation sets using stratified sampling
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Combine the X_train and y_train DataFrames back into a single DataFrame
        self.df_train = pd.concat([X_train, y_train], axis=1)

        # Combine the X_val and y_val DataFrames back into a single DataFrame
        self.df_val = pd.concat([X_val, y_val], axis=1)

    def train_dataloader(self):
        x = []
        y = []
        for text, c in zip(self.df_train['text'], self.df_train['class']):
            x_encoded = []
            for char in text:
                if char not in self.alphabet.keys():
                    x_encoded.append(len(self.alphabet))
                else:
                    x_encoded.append(self.alphabet.get(char,0))
            x_padded = x_encoded[:self.input_size] + [0] * (self.input_size - len(x_encoded))
            x.append(x_padded)
            y.append(self.class_to_id[c])
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=BATCH_SIZE
        )
    

    def val_dataloader(self):
        x = []
        y = []
        for text, c in zip(self.df_val['text'], self.df_val['class']):
            x_encoded = []
            for char in text:
                if char not in self.alphabet.keys():
                    x_encoded.append(len(self.alphabet))
                else:
                    x_encoded.append(self.alphabet.get(char,0))
            x_padded = x_encoded[:self.input_size] + [0] * (self.input_size - len(x_encoded))
            x.append(x_padded)
            y.append(self.class_to_id[c])
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=BATCH_SIZE
        )
    
    def one_hot_encode(self, x):
        x = x.cpu().numpy()
        x_one_hot = np.zeros((len(x), self.input_size, len(self.alphabet)), dtype=np.float32)
        # print("one hot shape ",x_one_hot.shape)
        for i, sentence in enumerate(x):
            for j, char_id in enumerate(sentence.tolist()):
                if j < self.input_size:
                    x_one_hot[i, j, char_id-1] = 1
        return torch.from_numpy(x_one_hot).to(self.device)
# Define the early stopping and checkpoint callbacks
early_stop_callback = EarlyStopping(
    monitor='val_acc',
    min_delta=0.00,
    patience=5,
    verbose=False,
    mode='max'
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath='checkpoints',
    filename='CharCNN-{epoch:02d}-{val_acc:.2f}',
    save_top_k=1,
    mode='max'
)

total_alphabet = "".join(set(latin_alphabet + symbols + greek_alphabet + found_list))
alphabet = {}
for char in total_alphabet:
    alphabet[char]=len(alphabet)
alphabet["<unk>"] =len(alphabet)
print(alphabet)
model = CharCNN(input_size=100, num_classes=14, alphabet=alphabet)
summary = ModelSummary(model)
print(summary)
early_stop_callback = EarlyStopping(monitor='val_acc', min_delta=0.00, patience=5, mode='max')
checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')
trainer = pl.Trainer(callbacks=[early_stop_callback, checkpoint_callback], max_epochs=100,accelerator="gpu", devices=1)
trainer.fit(model)