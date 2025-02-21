from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__, static_folder="static")

# Load the trained model
with open("E:/CODING/PROJECT/ML_Model_Deployment/backend/model.pkl", "rb") as f:
    model = pickle.load(f)

# Class labels
class_labels = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]  # Get the predicted class index
    class_name = class_labels[prediction]  # Convert index to class name
    
    return jsonify({"prediction": int(prediction), "flower": class_name})

if __name__ == '__main__':
    app.run(debug=True)
