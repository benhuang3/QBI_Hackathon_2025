from flask import Flask, render_template, request

app = Flask(__name__)

def model(pInt, dInt):
    return pInt + dInt


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None

    if request.method == 'POST':
        pseq = request.form.get('pseq')
        dstr = request.form.get('dstr')
        result = model(pseq, dstr)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)