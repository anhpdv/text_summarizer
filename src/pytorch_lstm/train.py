import os
import random
import numpy as np
import pandas as pd
import re
import warnings
from sklearn.model_selection import train_test_split
from underthesea import sent_tokenize, word_tokenize, text_normalize
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchnlp.encoders.text import (
    StaticTokenizerEncoder,
    stack_and_pad_tensors,
    pad_tensor,
)
from torchtext.data import Field, BucketIterator

try:
    from config.config import DEVICE, STOPWORDS_USE
    from src.pytorch_lstm.helpers import check_rate_word, clean_text, preprocess_text
    from src.pytorch_lstm.module import Decoder, Encoder, Seq2Seq

except ImportError:
    from helpers import add_path_init, check_rate_word, clean_text, preprocess_text

    add_path_init()
    from config import DEVICE, STOPWORDS_USE
    from pytorch_lstm.module import Decoder, Encoder, Seq2Seq


# Set random seed for reproducibility
torch.manual_seed(0)

# Ignore warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 200)


def load_data(data_link):
    if os.path.exists(data_link):
        data_train_test = pd.read_csv(data_link)
        print("Exit data preprocess")
    else:
        print("Start data preprocess")
        data_text_summary = pd.read_csv("./dataset/data_text_summary.csv")
        data_train_test = pd.DataFrame()
        data_train_test["summary"] = data_text_summary["summary"].apply(
            lambda x: preprocess_text(x, STOPWORDS_USE)
        )
        data_train_test["fulltext"] = data_text_summary["fulltext"].apply(
            lambda x: preprocess_text(x, STOPWORDS_USE)
        )
        data_train_test.to_csv(data_link, encoding="utf-8")

    return data_train_test


import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(
        self, input_size, embedding_dim, hidden_dim, num_layers=3, dropout=0.4
    ):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, input):
        embedded = self.embedding(input)
        output, (hidden, cell) = self.rnn(embedded)
        return output, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, encoder_outputs, decoder_hidden):
        attn_scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        attn_weights = nn.functional.softmax(attn_scores, dim=1).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(
        self,
        output_size,
        embedding_dim,
        hidden_dim,
        attention,
        num_layers=3,
        dropout=0.4,
    ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.attention = attention
        self.rnn = nn.LSTM(
            embedding_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        context, attn_weights = self.attention(encoder_outputs, hidden[-1])
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = output.squeeze(1)
        prediction = self.fc(output)
        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src, trg = batch.text, batch.summary
        optimizer.zero_grad()
        output = model(src, trg)
        # Trimming the output and target tensors to remove the <start> token
        output = output[:, 1:].reshape(-1, output.shape[2])
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg = batch.text, batch.summary
            output = model(src, trg, 0)  # Turn off teacher forcing
            # Trimming the output and target tensors to remove the <start> token
            output = output[:, 1:].reshape(-1, output.shape[2])
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def main():
    # Bước 1: Tiền xử lý dữ liệu
    data_link = "./dataset/data_train_test.csv"
    data_train_test = load_data(data_link)

    max_summary_len = 80
    max_text_len = 3000
    cleaned_text = np.array(data_train_test["fulltext"])
    cleaned_summary = np.array(data_train_test["summary"])

    short_text = []
    short_summary = []
    for i in range(len(cleaned_text)):
        if (
            len(cleaned_summary[i].split()) <= max_summary_len
            and len(cleaned_text[i].split()) <= max_text_len
        ):
            short_text.append(cleaned_text[i])
            short_summary.append(cleaned_summary[i])
    df = pd.DataFrame({"text": short_text, "summary": short_summary})
    # Thêm **START** và **END** tokens vào 2 đầu của summary (**start** - start of summary token, **end** - end of summary token)
    df["summary"] = df["summary"].apply(lambda x: "start " + x + " end")
    # Split the data into training and validation sets
    train_data, valid_data = train_test_split(
        df, test_size=0.2, random_state=0, shuffle=True
    )

    # Step 2: Define Fields
    TEXT = Field(tokenize="spacy", init_token="<sos>", eos_token="<eos>", lower=True)
    SUMMARY = Field(tokenize="spacy", init_token="<sos>", eos_token="<eos>", lower=True)

    # Step 3: Build Vocabulary
    TEXT.build_vocab(train_data, max_size=10000, min_freq=2)
    SUMMARY.build_vocab(train_data, max_size=10000, min_freq=2)

    # Step 4: Create Iterators
    BATCH_SIZE = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_iterator, valid_iterator = BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
        device=device,
    )

    # Step 5: Define Model, Optimizer, and Loss Function
    INPUT_DIM = len(TEXT.vocab)
    OUTPUT_DIM = len(SUMMARY.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    LEARNING_RATE = 0.0005
    CLIP = 1

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    attn = Attention(HID_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, attn, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=SUMMARY.vocab.stoi[SUMMARY.pad_token])

    # Step 6: Training Loop
    N_EPOCHS = 10
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
        print(
            f"Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}"
        )


if __name__ == "__main__":
    main()
