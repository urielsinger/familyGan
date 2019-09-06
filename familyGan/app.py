from flask import *
import base64
import os
from pipeline import integrate_with_web
app = Flask(__name__)

UPLOAD_FOLDER = 'upload_files'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_base64_image(path):
    encoded_string = ""
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#######################  Get Web Page #####################

@app.route("/")
def home(name=None):
    return render_template('index.html', name=name)

@app.route("/<imgname>.png")
def result(imgname=None):
    return render_template('result.html', name=imgname)

#######################  Get Images #####################

@app.route('/get_image/<img_name>')
def get_image(img_name):
    if img_name and allowed_file(img_name):
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], img_name), mimetype='image/png')

@app.route('/get_image_base64/<img_name>')
def get_image_base64(img_name):
    if img_name and allowed_file(img_name):
        return get_base64_image(os.path.join(app.config['UPLOAD_FOLDER'], img_name))

#######################  Upload File #####################33

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image1 = request.files['image1']
        image2 = request.files['image2']
        if image1 and allowed_file(image1.filename) and image2 and allowed_file(image2.filename):
            father_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'], image1.filename)
            mother_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'], image2.filename)

            image1.save(father_path)
            image2.save(mother_path)

            child_image_name = integrate_with_web(father_path,mother_path)

            return child_image_name
        else:
            return 'File not allowed'
    return False

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)