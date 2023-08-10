#import required libraries
from flask import Flask, render_template, request 
import joblib 
import numpy as np

#instantiate the Flask app 
app = Flask(__name__, template_folder = 'web') 

#define the route and response for accessing the home page
@app.route('/') 
def home_page(): 
    return render_template('home.html') 

#define the function that loads model and does prediction
def prediction_service(input_values): 
    to_predict = np.array(input_values).reshape(-1,1)
    loaded_model = joblib.load('model.sav') 
    result = loaded_model.predict(to_predict) 
    return result[0] 

'''define the route for form submission i.e., for passing
in the input values to the web app to get predicted values''' 
@app.route('/', methods = ['POST'])
def predicted_value(): 
    if request.method == 'POST': 
        input_values = request.form.to_dict() 
        input_values = list(input_values.values()) 
        to_predict = list(map(float, input_values)) 
        result = round(float(prediction_service(to_predict), 2)) 
        return render_template('home.html', result = result) 
    
if __name__ == '__main__': 
    app.run(debug = True)



