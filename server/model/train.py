import random
import string
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# nltk.download()

IGNORED_LETTERS = [p for p in string.punctuation]

lemmatizer = WordNetLemmatizer()


def train():
    """
    Prepare data,
    save words/classes,
    train model,
    save model.

    :return: None
    """
    # PREPARE DATA
    data = json.loads(open(f'data/database_{lang}.json').read())
    classes = []
    for item in data:
        if item['tag'] not in classes:
            classes.append(item['tag'])
    classes = sorted(classes)
    words = []
    for item in data:
        for pattern in item['patterns']:
            words.extend(nltk.word_tokenize(pattern))
    words = [lemmatizer.lemmatize(word) for word in words if word not in IGNORED_LETTERS]
    words = sorted(set(words))
    documents = []
    for item in data:
        for pattern in item['patterns']:
            documents.append((nltk.word_tokenize(pattern), item['tag']))

    training = []

    for document in documents:
        input_representation = np.zeros(len(words))
        output_representation = np.zeros(len(classes))
        for word in words:
            if word in document[0]:
                input_representation[words.index(word)] = 1
        output_representation[classes.index(document[1])] = 1
        training.append([input_representation, output_representation])

    random.shuffle(training)
    training = np.array(training)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    # SAVE WORDS/CLASSES
    pickle.dump(words, open(f'output/words_{lang}.pkl', 'wb'))
    pickle.dump(classes, open(f'output/classes_{lang}.pkl', 'wb'))

    # CREATE/TRAIN/SAVE MODEL
    model = Sequential()

    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(learning_rate=0.01)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    result = model.fit(x=np.array(train_x), y=np.array(train_y), batch_size=4, epochs=200, verbose=1)
    model.save(f'output/model_{lang}.h5', result)


languages = ['en']  # TODO ADD LATER 'fr', 'ar'

for lang in languages:
    train()
