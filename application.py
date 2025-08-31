import pickle
from flask import Flask, request, render_template


app = Flask(__name__)

# Load Ridge model and StandardScaler
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
Standard_Scaler = pickle.load(open("models/scaler.pkl", "rb"))


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoints():
    if request.method=="POST":
        # convert all inputs to float
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))   # this was missing earlier
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        # scale and predict
        new_data_scaled = Standard_Scaler.transform(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )
        result = ridge_model.predict(new_data_scaled)

        return render_template("home.html", results=result[0])
    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
