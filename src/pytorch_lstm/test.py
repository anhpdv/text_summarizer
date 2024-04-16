import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

# Dữ liệu mẫu
data = [
    ("Đây là câu gốc.", "Đây là tóm tắt."),
    ("Một câu ví dụ khác.", "Tóm tắt ví dụ."),
    # Thêm dữ liệu khác nếu cần
]


# Chuyển đổi văn bản thành tensor
def text_to_tensor(text, vocab):
    tensor = [vocab[word] for word in text]
    return torch.tensor(tensor, dtype=torch.long)


# Xây dựng từ vựng
vocab = {}
for pair in data:
    for sentence in pair:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)


# Xây dựng mô hình LSTM
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True
        )  # Set batch_first=True
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, _ = self.lstm(embedded)  # Remove unsqueeze(1)
        output = self.linear(output)
        return output


# Khởi tạo mô hình
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
model = LSTMModel(vocab_size, embedding_dim, hidden_dim)

# Hàm mất mát và tối ưu hóa
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

# Training loop
# Training loop
batch_size = 2  # Set your desired batch size
for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(0, len(data), batch_size):  # Process data in batches
        batch = data[i : i + batch_size]

        # Initialize lists to store batch input and target tensors
        input_tensors = []
        target_tensors = []

        # Find the maximum length in the batch for padding
        max_input_len = max(len(pair[0].split()) for pair in batch)
        max_target_len = max(len(pair[1].split()) for pair in batch)

        # Convert batch data to tensors and pad sequences
        for pair in batch:
            input_tensor = text_to_tensor(pair[0].split(), vocab)
            target_tensor = text_to_tensor(pair[1].split(), vocab)
            # Pad input sequence
            input_pad_len = max_input_len - len(pair[0].split())
            input_tensor = nn.functional.pad(input_tensor, (0, input_pad_len), value=0)
            input_tensors.append(input_tensor.unsqueeze(0))  # Add batch dimension
            # Pad target sequence
            target_pad_len = max_target_len - len(pair[1].split())
            target_tensor = nn.functional.pad(
                target_tensor, (0, target_pad_len), value=0
            )
            target_tensors.append(target_tensor.unsqueeze(0))  # Add batch dimension

        # Stack input and target tensors into batch tensors
        input_tensors = torch.cat(
            input_tensors, dim=0
        )  # Concatenate along batch dimension
        target_tensors = torch.cat(
            target_tensors, dim=0
        )  # Concatenate along batch dimension

        optimizer.zero_grad()

        output = model(input_tensors)

        # Flatten the output and target tensors to calculate loss
        loss = criterion(output.view(-1, vocab_size), target_tensors.view(-1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / (len(data) / batch_size)}"
    )
