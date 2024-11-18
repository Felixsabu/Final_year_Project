from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import os
import pickle 
import cv2
from matplotlib import pyplot as plt

app = Flask(__name__)

# Set the path for the current directory
web = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained model
with open('Model_SVM_C2.pkl', 'rb') as pickle_file:
    new_data = pickle.load(pickle_file)

# Routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("appindex.html")

@app.route("/about")
def about_page():
    return "Please subscribe to Artificial Intelligence Hub..!!!"

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        # Retrieve the image file from the form
        imagefile = request.files['my_image']
        image_path = "static/" + imagefile.filename
        imagefile.save(image_path)

        # Load the image
        img = cv2.imread(image_path)
        height, width, channels = img.shape

        # Proportional slicing based on image size
        # Example: Extract a region that is 5% of the width and height starting at 80% down and across the image
        start_x = int(0.8 * width)
        end_x = start_x + int(0.1 * width)
        start_y = int(0.8 * height)
        end_y = start_y + int(0.1 * height)

        # Ensure the slicing is within bounds
        if end_x <= width and end_y <= height:
            S5RGB_imgA1A1_1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract regions dynamically based on the computed positions
            S5meanrefA1_1 = S5RGB_imgA1A1_1[start_y:end_y, start_x:end_x]
            S5meanA1_2 = S5RGB_imgA1A1_1[start_y:end_y, start_x + int(0.15 * width):start_x + int(0.25 * width)]

            # Reshape the slices
            S5refA1_1 = np.reshape(S5meanrefA1_1, (-1, 3))
            S5refA1_2 = np.reshape(S5meanA1_2, (-1, 3))

            # Calculate mean values for RGB channels
            Rref = S5refA1_1[:, 0].mean()
            Gref = S5refA1_1[:, 1].mean()
            Bref = S5refA1_1[:, 2].mean()
            R = S5refA1_2[:, 0].mean()
            G = S5refA1_2[:, 1].mean()
            B = S5refA1_2[:, 2].mean()

            # Prepare input for the model
            list_total = [Rref, Gref, Bref, R, G, B]
            list_total2 = [list_total]

            # Predict using the pre-trained model
            predic_model = new_data.predict(list_total2)
            model = predic_model[0]

            # Load the CSV file and extract the prediction details
            out_model = pd.read_csv('OUTPUT.csv')
            test = out_model[out_model['ชื่อเฉดไกด์'] == model]

            # Extract the prediction details from the CSV
            Dict1 = str(test['ชื่อเฉดไกด์'].iloc[0])
            Dict2 = str(test['ค่าสี CIE l a b'].iloc[0])
            Dict3 = str(test['โทนสี'].iloc[0])
            Dict4 = str(test['ชื่อเฉดไกด์ที่ใกล้เคียง'].iloc[0])
            Dict5 = str(test['เทียบเท่าเฉดไกด์ 3D Master'].iloc[0])

            # Return the result to the template
            return render_template(
                "appindex.html",
                test1=Dict1,
                test2=Dict2,
                test3=Dict3,
                test4=Dict4,
                test5=Dict5,
                image_target=image_path
            )
        else:
            return "Image size is too small for the required processing."

@app.route('/display/<image_target>')
def display_image(image_target):
    return redirect(url_for('static', filename=image_target), code=301)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

