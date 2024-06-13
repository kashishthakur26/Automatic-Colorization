from flask import Flask, request, render_template, send_file
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import torch


from src import eccv16, siggraph17, load_img, preprocess_img, postprocess_tens

app = Flask(__name__)


#loading model
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

#GPu availablity
use_gpu = torch.cuda.is_available()
if use_gpu:
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not  in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        img = Image.open(file)
        img = np.array(img)

        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
        if use_gpu:
            tens_l_rs = tens_l_rs.cuda()

        out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
        out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
        buffer = BytesIO()
        Image.fromarray((out_img_eccv16 * 255).astype(np.uint8)).save(buffer, format="PNG")
        buffer.seek(0)  # Rewind the buffer to the beginning

        # Send the image as a response
        return send_file(buffer, mimetype='image/png')
    
if __name__ == '__main__':
    app.run(debug=True)
        