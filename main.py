import pandas as pd
from bs4 import BeautifulSoup
import string
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dataset = pd.read_csv("Musical_instruments_reviews.csv")
dataset = dataset[['reviewText','overall','summary']]
print('\n[Overall Distribution]')
print(dataset['overall'].value_counts(normalize=True))
print('\n[Missing Values Counts]')
print(dataset.isna().sum())
print('\n[Number of rows]')
print(len(dataset))


### DROPPING MISSING VALUES
dataset.dropna(inplace=True)
print('[DATASET AFTER DROPPING MISSING VALUES OUT]')
print(dataset.head())
print('\n[Overall Distribution]')
print(dataset['overall'].value_counts(normalize=True))
print('\n[Missing Values Counts]')
print(dataset.isna().sum())
print('\n[Number of rows]')
print(len(dataset))


### PUTTING TOGETHER SUMMARY AND TEXT
dataset['sentences'] = dataset['reviewText'] + ' ' + dataset['summary']
dataset['label'] = dataset['overall']
dataset = dataset[['sentences','label']]
print(dataset.head(5))

### PREPROCESSING DATA
#Removing Stop Words and Removing punctuation
stopwords = []
with open('englishST.txt', encoding='UTF-8') as f:
    for line in f:
        stopwords.append(line[:-1])
print(stopwords)

preprocessed_sentences = []
for sentence in dataset['sentences']:
    sentence = sentence.replace(',',' , ')
    sentence = sentence.replace('-',' - ')
    sentence = sentence.replace('.',' . ')
    sentence = sentence.replace('/',' / ')
    table = str.maketrans('', '', string.punctuation)
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ''
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + ' ' + word
    filtered_sentence = filtered_sentence.replace('  ',' ')
    preprocessed_sentences.append(filtered_sentence)

dataset['sentences'] = preprocessed_sentences
print(dataset.head())

### Plotting the length of each sentence

xs = []
ys = []
current_item = 1
for item in list(dataset['sentences']):
    xs.append(current_item)
    current_item +=1
    ys.append(len(item))
newys = sorted(ys)
plt.plot(xs,newys)
plt.show()

dataset = dataset.sample(frac=1)
TRAINING_SIZE = int(len(dataset)*0.8)
VALIDATION_SIZE = int(len(dataset)*0.2)
labels= np.array(dataset['label'])

for i in range(0,len(labels)):
    if labels[i]==5 or labels[i]==4:
        labels[i]=1
    else:
        labels[i]=0

training_sentences = np.array(dataset['sentences'][:TRAINING_SIZE])
training_label = np.array(labels[:TRAINING_SIZE])
validation_sentences = np.array(dataset['sentences'][TRAINING_SIZE:-VALIDATION_SIZE])
validation_label = np.array(labels[TRAINING_SIZE:-VALIDATION_SIZE])
testing_sentences = np.array(dataset['sentences'][-VALIDATION_SIZE:])
testing_label = np.array(labels[-VALIDATION_SIZE:])




vocab_size = 20000
max_length = 1000
trunc_type = 'post'
padding_type = 'post'
oov_token = '<OOV>'

tokenizer = Tokenizer(oov_token=oov_token,num_words=vocab_size)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences,
                                maxlen= max_length,
                                padding=padding_type,
                                truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences,
                                maxlen= max_length,
                                padding=padding_type,
                                truncating=trunc_type)



print(training_label)

#training_label = tf.one_hot(training_label,5)
#fourth root of vocab_size
embedding_dim = int(vocab_size**0.25)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
#print(training_padded[:5])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(np.array(training_padded),
                    training_label,
                    validation_data=(validation_padded,validation_label),
                    epochs=50)

model.evaluate(pad_sequences(tokenizer.texts_to_sequences(testing_sentences),
                                                            maxlen=max_length,
                                                            padding=padding_type,
                                                            truncating=trunc_type),
                                                            testing_label)


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(accuracy)
plt.plot(val_accuracy)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Accuracy comparison')
plt.show()

plt.plot(loss)
plt.plot(val_loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Loss comparison')
plt.show()

