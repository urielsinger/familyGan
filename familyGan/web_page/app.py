from flask import *
import base64
import os
from datetime import datetime, timedelta
from math import ceil
from familyGan.pipeline import integrate_with_web_get_child, integrate_with_web_get_generated_child
app = Flask(__name__)

UPLOAD_FOLDER = 'upload_files'
CHILD_FOLDER = 'generated_files'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CHILD_FOLDER'] = CHILD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

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

@app.route('/get_child_image/<img_name>')
def get_child_image(img_name):
    if img_name and allowed_file(img_name):
        return send_file(os.path.join(app.config['CHILD_FOLDER'], img_name), mimetype='image/png')

@app.route('/get_image_base64/<img_name>')
def get_image_base64(img_name):
    if img_name and allowed_file(img_name):
        return get_base64_image(os.path.join(app.config['UPLOAD_FOLDER'], img_name))

#######################  Upload File #####################33

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    def generate():
        if request.method == 'POST':
            image1 = request.files['image1']
            image2 = request.files['image2']
            if image1 and allowed_file(image1.filename) and image2 and allowed_file(image2.filename):
                father_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], image1.filename)
                mother_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], image2.filename)

                image1.save(father_path)
                image2.save(mother_path)

                gen = integrate_with_web_get_child(father_path, mother_path)
                last_time = datetime.now()
                child_image_name = next(gen)
                while isinstance(child_image_name, float):

                    cur_time = datetime.now()
                    delta = cur_time - last_time
                    pred_time = 'Inf'
                    if child_image_name > 0:
                        pred_time = timedelta(seconds=ceil(delta.total_seconds()/child_image_name))
                    delta = timedelta(seconds=ceil(delta.total_seconds()))
                    percent = round(100*child_image_name, 2)
                    yield {'width': str(percent),
                           'time_passed': delta.__str__(),
                           'time_estimation': pred_time.__str__()}.__str__().replace('\'',"\"")
                    child_image_name = next(gen)

                yield child_image_name
            else:
                yield 'File not allowed'
        else:
            yield False
    return Response(stream_with_context(generate()))

@app.route('/generate', methods=['GET', 'POST'])
def generate_child():
    if request.method == 'POST':

        kwargs = dict(request.form)

        child_latent_path = kwargs.pop('child_path')[:-3] + 'npy'
        child_latent_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['CHILD_FOLDER'], child_latent_path)

        kwargs = {k: float(v) for k, v in kwargs.items() if float(v) != 0}

        child_image_name = integrate_with_web_get_generated_child(child_latent_path, **kwargs)

        return child_image_name

    return False

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)