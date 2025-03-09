from flask import Flask, render_template, request
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app = Flask(__name__)

def model(pInt, dInt):
    return str(pInt) + str(dInt)

def input_to_list(input_string: str) -> list:
    return [value.strip() for value in re.split(r'[\s,]+', input_string) if value.strip()]


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None

    if request.method == 'POST':
        pseq = request.form.get('pseq')
        dstr = request.form.get('dstr')

        pseq = input_to_list(pseq)
        dstr = input_to_list(dstr)
        if len(pseq) != len(dstr):
            result = "Number of Values Do Not Match"
        else:
            result = []
            for i,j in zip(pseq, dstr):
                result.append(model(i, j))





    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)