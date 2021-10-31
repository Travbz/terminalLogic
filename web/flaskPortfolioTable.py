import pandas as pd
from flask import Flask, render_template
from flask_cors import CORS


data = pd.read_csv("./templates/portfolio2.json")

app = Flask(__name__)
CORS(app)
headings = data.columns
@app.route("/")
def table():
    return render_template("portfolio.html")

if __name__=="__main__":
    app.run(debug=True)
