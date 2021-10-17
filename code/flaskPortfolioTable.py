import pandas as pd
from flask import Flask, render_template

data = pd.read_csv("../web/assets/portfolio.csv")
app = Flask(__name__)
headings = data.columns
@app.route("/")
def table():
    return render_template("./web/templates/portfolio.html", headings=headings, data=data)

if __name__=="__main__":
    app.run(debug=True)
