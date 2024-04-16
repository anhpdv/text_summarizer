import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence


# Định nghĩa mô hình Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input):
        packed_input = input[0]  # Packed sequence
        input_lengths = input[1]  # Lengths of sequences

        embedded = self.embedding(packed_input)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.lstm(packed_embedded)
        return output, hidden



# Định nghĩa mô hình Attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        attn_energies = torch.zeros(seq_len)
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
        return torch.softmax(attn_energies, dim=0)

    def score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden[0], encoder_output), 1))
        energy = torch.dot(self.v, energy)
        return energy


# Định nghĩa mô hình Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.attn = Attention(self.hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        attn_weights = self.attn(hidden[0], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        lstm_input = torch.cat((embedded, context), 2)
        output, hidden = self.lstm(lstm_input, hidden)
        output = torch.softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights


# Hàm huấn luyện
def train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    max_length=10,
):
    encoder_hidden = torch.zeros(1, 1, encoder.hidden_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Assuming input_tensor is already padded
    encoder_output, encoder_hidden = encoder(input_tensor)

    decoder_input = torch.tensor([[0]])  # Assuming you have a start token with index 0
    decoder_hidden = encoder_hidden

    loss = 0

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output
        )
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == 1:  # Assuming you have an end token with index 1
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# Định nghĩa lớp tập dữ liệu
class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_text = self.data.iloc[idx]["fulltext"]
        summary = self.data.iloc[idx]["summary"]
        return {"full_text": full_text, "summary": summary}


# Chuẩn bị dữ liệu
dataset = CustomDataset("./dataset/data_train_test.csv")

# Tạo một DataLoader để lặp lại qua dữ liệu
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
sample = dataset[0]
# print("Full Text:", sample["full_text"])
# print("Summary:", sample["summary"])

# Tham số mô hình
input_size = 10
hidden_size = 8
output_size = 10
learning_rate = 0.01
max_length = 10

# Khởi tạo mô hình và bộ tối ưu
encoder = Encoder(input_size, hidden_size)
decoder = Decoder(hidden_size, output_size)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


def text_to_tensor(text, word_to_index):
    # Chuyển đổi mỗi từ trong văn bản thành chỉ số trong từ điển
    indexes = [word_to_index[word] for word in text.split()]
    # Tạo tensor từ các chỉ số
    tensor = torch.tensor(indexes, dtype=torch.long)
    return tensor


# Tạo từ điển từ văn bản
word_to_index = {}
for text in dataset.data["fulltext"]:
    for word in text.split():
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

# Thêm một token đặc biệt cho việc padding
word_to_index["<PAD>"] = len(word_to_index)

# Huấn luyện mô hình
for epoch in range(100):
    total_loss = 0
    for batch in data_loader:
        full_texts = batch["full_text"]
        summaries = batch["summary"]

        # Khởi tạo optimizer cho mỗi vòng lặp
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Mã hóa full text
        full_texts_tensor = pad_sequence(
            [text_to_tensor(full_text, word_to_index) for full_text in full_texts]
        )
        # Tính toán độ dài thực tế của mỗi mẫu trong batch
        input_lengths = torch.tensor([len(text.split()) for text in full_texts])
        # Sắp xếp dữ liệu theo độ dài giảm dần
        input_lengths, sorted_indices = input_lengths.sort(descending=True)
        full_texts_tensor = full_texts_tensor[:, sorted_indices]
        # Truyền dữ liệu qua encoder
        packed_input = pack_padded_sequence(full_texts_tensor, input_lengths)
        encoder_output, encoder_hidden = encoder(packed_input)

        # Khởi tạo đầu vào của decoder
        decoder_input = torch.tensor([[0]] * len(full_texts))

        # Mã hóa summary
        decoder_hidden = encoder_hidden
        loss = 0
        for di in range(len(summaries)):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output
            )
            loss += criterion(decoder_output, summaries[di])
            # Sử dụng summary đã được sinh ra làm đầu vào cho bước tiếp theo
            decoder_input = summaries[di]

        # Tính toán và cập nhật gradients
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader.dataset):.4f}")
