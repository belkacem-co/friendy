import random
import json
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

ERROR_THRESHOLD = 0.25

lemmatizer = WordNetLemmatizer()


def predict_class(user_input, model, words, classes):
    """
    Takes the user input,
    transform it to numeric values,
    then feed it to the model,
    finally return each class with its respective probability.

    :param user_input: user's message
    :param model: loaded model
    :param words: list of words used while training the model
    :param classes: list of classes used while training the model
    :return: a dictionary in the format of { 'class': '', 'probability': '' }
    """
    input_words = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(user_input)]
    input_representation = np.zeros(len(words))
    for input_word in input_words:
        for index, word in enumerate(words):
            if input_word == word:
                input_representation[index] = 1
    probabilities = []
    for index, probability in enumerate(model.predict(np.array([input_representation]))[0]):
        if probability > ERROR_THRESHOLD:
            probabilities.append([index, probability])
    probabilities.sort(key=lambda x: x[1], reverse=True)
    result = []
    for probability in probabilities:
        result.append({
            'class': classes[probability[0]],
            'probability': str(probability[1])
        })
    return result


def generate_response(user_input, lang):
    """
    Loads model/data based on the user's preferred language,
    call @predict_class function,
    generate a response randomly using the 'tag' related to the highest probability.

    :param user_input: the user's message
    :param lang: the language chosen by the user
    :return: an object containing the response and the recommended prepositions
    """
    # LOAD NECESSARY DATA
    data = json.loads(open(f'model/data/database_{lang}.json').read())
    groups = json.loads(open(f'model/data/groups_{lang}.json').read())
    words = pickle.load(open(f'model/output/words_{lang}.pkl', 'rb'))
    classes = pickle.load(open(f'model/output/classes_{lang}.pkl', 'rb'))
    # LOAD MODEL
    model = load_model(f'model/output/model_{lang}.h5')
    # PREDICT CLASS
    predictions = predict_class(user_input, model, words, classes)
    response = 'Error'
    code = ''
    for item in data:
        if item['tag'] == predictions[0]['class']:
            response = random.choice(item['responses'])
            code = item['code']
            break

    group = None
    for item in groups:
        if item['code'] == code:
            group = item

    next_contexts = []
    for item in groups:
        if group:
            if item['code'] in group['to']:
                next_contexts.append(item)

    return {
        'response': response,
        'propositions': list(map(lambda x: x['proposition'], next_contexts))
    }
