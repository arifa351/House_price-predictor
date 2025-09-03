from flask import Flask , render_template , request
import pandas as pd
import pickle
app = Flask(__name__)
@app.route('/' , methods =['GET'])
def form():
    return render_template('index.html')
@app.route('/predict' , methods =['POST'])
def predict():
    rooms = request.form['Rooms']
    size = request.form['Size']
    

    # make a dataframe for model input
    house_df = pd.DataFrame([[rooms, size]], 
                            columns=['Rooms', 'Size'])
    with open('model1.pkl', 'rb') as obj:
        model = pickle.load(obj)
        prediction = model.predict(house_df)
        return render_template('index.html', prediction = prediction)
if __name__ == '__main__':
    app.run(debug=True) 