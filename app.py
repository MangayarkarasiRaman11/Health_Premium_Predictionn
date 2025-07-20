from flask import Flask, render_template, request, redirect, url_for, session
from functools import wraps
import numpy as np
import pickle

# Load Trained Model & Preprocessing Tools
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("label_encoders.pkl", "rb") as file:
    label_encoders = pickle.load(file)




app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session management

# Dummy User Database (Replace with a real database in production)
users = {}

# Authentication Decorator to Restrict Access to Prediction Page
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))  # Redirect to login page if not logged in
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required  # Restricting access
def predict():
    if request.method == 'GET':
        return render_template("index.html")
    try:
        # Get Input Values from Form
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        sex = label_encoders['sex'].transform([request.form['sex']])[0]
        smoker = label_encoders['smoker'].transform([request.form['smoker']])[0]
        region = label_encoders['region'].transform([request.form['region']])[0]

        # Ensure Feature Order Matches Training
        input_data = np.array([[age, bmi, children, sex, smoker, region]])
        input_data_scaled = scaler.transform(input_data)  # Standardize all features

        # Make Prediction
        prediction = model.predict(input_data_scaled)[0]

        return render_template("result.html", prediction=round(prediction, 2))

    except Exception as e:
        return render_template("result.html", error=str(e))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users and users[email] == password:
            session['user'] = email  # Store user session
            return redirect(url_for('predict'))  # Redirect to prediction page after login
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        if email in users:
            return render_template("register.html", error="Email already registered")
        users[email] = password
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route('/logout')
def logout():
    session.pop('user', None)  # Remove user session
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
