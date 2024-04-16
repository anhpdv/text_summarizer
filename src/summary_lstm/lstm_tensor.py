import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(128, input_shape=(max_len, embedding_dim), return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(vocabulary_size, activation='softmax'))

# Compile mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)

# Sử dụng mô hình để tạo tóm tắt
summary = model.predict(new_text)
