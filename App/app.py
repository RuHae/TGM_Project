# app.py
from flask import Flask, render_template  

app = Flask(__name__) # name for the Flask app (refer to output)

@app.route("/", methods=['GET']) # decorator
def home(): # route handler function
    # returning a response
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)