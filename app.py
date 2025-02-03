import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# TPU 연결
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("TPU 연결 성공!")
except:
    strategy = tf.distribute.get_strategy()
    print("TPU 없음, CPU/GPU 사용")

# 데이터 로딩 (업로드한 텍스트 파일 경로 맞춰야 함)
file_path = "/kaggle/input/your-dataset-name/hanguel.txt"  # 업로드한 파일 경로 맞춰야 함

with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# 토크나이저 설정
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# 시퀀스 생성 (중복 및 과도하게 많은 n-그램)
input_sequences = []
for line in text.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# 과도한 패딩 추가
max_sequence_len = max(len(seq) for seq in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# 모델 생성 (268,435,456차원 임베딩 및 LSTM 설정)
with strategy.scope():
    model = Sequential([
        Embedding(total_words, 268435456, input_length=max_sequence_len-1),  # 268,435,456차원 임베딩
        LSTM(268435456, return_sequences=True),  # LSTM 레이어 268,435,456 유닛
        LSTM(268435456),
        Dense(268435456, activation='relu'),  # 268,435,456 크기 Dense 레이어
        Dense(total_words, activation='softmax')  # 최종 출력
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 자원 낭비를 위해 학습
for epoch in range(268435456):  # 268,435,456번 학습
    print(f"Epoch {epoch+1}/268435456")
    model.fit(X, y, epochs=1, verbose=1)

# 모델 저장
model.save('/kaggle/working/my_model_good.h5')

# 텍스트 생성 함수
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        output_word = tokenizer.index_word.get(predicted_word_index, "")
        seed_text += " " + output_word
    return seed_text

# 생성 예시
print(generate_text("안녕하세요", 50))  # 50단어 이상 생성
