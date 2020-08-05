#!/usr/bin/python
# Output @ http://localhost:5000
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
# import pickle
import os
import csv
import sys
import subprocess



# data_dir=os.path.abspath(os.path.join(os.path.curdir,"tmp"))
# checkpoint_dir=os.path.abspath(os.path.join(os.path.curdir,"runs","1594130120"))
# checkpoint_dir=os.path.abspath(os.path.join(os.path.curdir,"trainer","runs/1594130120/checkpoints/"))
# out_path=os.path.join(os.path.curdir,"online_prediction")
# bucket="gs://ordinal-reason-282519-aiplatform/text_cnn_training_071320201841"

app = Flask(__name__)
# model = pickle.load(open('randomForestRegressor.pkl','rb'))

# Prepare to receive reviews and write to file
if os.path.exists(os.path.join("/tmp/","input_reviews.csv")):
#    open(os.path.join("/tmp/","input_reviews.csv"))
   os.remove(os.path.join("/tmp/","input_reviews.csv"))
   print("Review File Removed")
else:
   open(os.path.join("/tmp/","input_reviews.csv"), 'w').close()
   print("Review File Created")
   
@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = request.form['Review']
    print(" Display reviews : {}".format(int_features))
    with open(os.path.join("/tmp/","input_reviews.csv"), 'w') as f:
    	 csv.writer(f).writerows(np.column_stack([int_features]))
#          fhandler.write("\tmp\{}".format("input_reviews.csv"))
    
#     prediction = os.system("python predict.py --eval_train") 
    subprocess.check_output([sys.executable,"predict.py","--eval_train"])

    with open(os.path.join("/tmp/","online_prediction.csv"), 'r') as f:
    	 prediction=np.column_stack(list(csv.reader(f)))
    print ("Prediction: {}".format(prediction))
    review_text="Review Text: {}".format(prediction[0][0])
    sentiment="Predicted Sentiment: {}".format(prediction[1][0])
    return render_template('home.html', **locals())

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)