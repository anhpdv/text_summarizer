import random
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from config.config import DEVICE
    from src.pytorch_lstm.helpers import preprocess_text

except ImportError:
    from helpers import add_path_init, preprocess_text

    add_path_init()
    from config import DEVICE


# Định nghĩa Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(
            1, 1, self.hidden_size, device=DEVICE
        )  # Sử dụng DEVICE ở đây


# Định nghĩa Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(
            1, 1, self.hidden_size, device=DEVICE
        )  # Sử dụng DEVICE ở đây


# Xây dựng từ điển
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Định nghĩa token bắt đầu (Start Of Sentence)
SOS_token = 0

# Định nghĩa token kết thúc (End Of Sentence)
EOS_token = 1


def save_checkpoint(
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    input_lang,
    output_lang,
    filename,
):
    state = {
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "encoder_optimizer_state_dict": encoder_optimizer.state_dict(),
        "decoder_optimizer_state_dict": decoder_optimizer.state_dict(),
        "input_lang": input_lang,
        "output_lang": output_lang,
    }
    torch.save(state, filename)


def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(" ")]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def load_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, filename):
    state = torch.load(filename)
    encoder.load_state_dict(state["encoder_state_dict"])
    decoder.load_state_dict(state["decoder_state_dict"])
    encoder_optimizer.load_state_dict(state["encoder_optimizer_state_dict"])
    decoder_optimizer.load_state_dict(state["decoder_optimizer_state_dict"])
    input_lang = state["input_lang"]
    output_lang = state["output_lang"]
    return input_lang, output_lang


def load_model(encoder, decoder, encoder_optimizer, decoder_optimizer, filename):
    state = torch.load(filename)
    encoder.load_state_dict(state["encoder_state_dict"])
    decoder.load_state_dict(state["decoder_state_dict"])
    encoder_optimizer.load_state_dict(state["encoder_optimizer_state_dict"])
    decoder_optimizer.load_state_dict(state["decoder_optimizer_state_dict"])


def main():

    data_link = "./dataset/data_train_test.csv"
    df = pd.read_csv(data_link)

    # print(df)
    df["fulltext"] = df["fulltext"].apply(preprocess_text)
    df["summary"] = df["summary"].apply(preprocess_text)

    input_lang = Lang("fulltext")
    output_lang = Lang("summary")

    for index, row in df.iterrows():
        input_lang.addSentence(row["fulltext"])
        output_lang.addSentence(row["summary"])

    # Định nghĩa tham số
    hidden_size = 256

    # Khởi tạo mô hình và bộ tối ưu hóa
    print("Load mô hình")
    encoder = Encoder(input_lang.n_words, hidden_size).to(DEVICE)
    decoder = Decoder(hidden_size, output_lang.n_words).to(DEVICE)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.01)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.01)

    # Load the model
    print("Loading the model...")
    if load_model(
        encoder, decoder, encoder_optimizer, decoder_optimizer, "seq2seq_model.pth"
    ):
        print("Mô hình đã được load thành công.")
    else:
        print("Không thể load mô hình.")
        return

    # Lấy một văn bản đầu tiên từ dataframe để test mô hình
    test_text = df.loc[0, "fulltext"]
    print("Văn bản đầu tiên từ dataframe:", test_text)

    # Tiền xử lý văn bản đầu tiên
    preprocessed_text = preprocess_text(test_text)
    MAX_LENGTH = 80
    # Chuyển đổi văn bản thành tensor
    input_tensor = tensorFromSentence(input_lang, preprocessed_text)

    # Dự đoán đầu ra sử dụng mô hình đã được load
    with torch.no_grad():
        encoder_hidden = encoder.initHidden().to(DEVICE)
        for ei in range(input_tensor.size(0)):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei].to(DEVICE), encoder_hidden
            )

        decoder_input = torch.tensor([[SOS_token]], device=DEVICE)
        decoder_hidden = encoder_hidden
        decoded_words = []
        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        decoded_sentence = " ".join(decoded_words)
        print("Đầu ra từ mô hình:", decoded_sentence)


if __name__ == "__main__":
    main()
