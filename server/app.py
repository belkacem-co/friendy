from flask import Flask
from flask import request

from model.use import generate_response

app = Flask(__name__)


@app.route('/')
def response():
    lang = request.headers.get('lang')
    user_input = request.headers.get('user-input')
    return generate_response(user_input=user_input, lang=lang)


if __name__ == '__main__':
    app.run()
