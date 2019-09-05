from flask import *
import os
from werkzeug import secure_filename

app = Flask(__name__)

# searchword = request.args.get('q', '')

UPLOAD_FOLDER = 'upload_files'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home(name=None):
    return render_template('index.html', name=name)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image1 = request.files['image1']
        image2 = request.files['image2']
        if image1 and allowed_file(image1.filename) and image2 and allowed_file(image2.filename):
            #            filename = secure_filename(file.filename)
            image1.save(os.path.join(app.config['UPLOAD_FOLDER'], image1.filename))
            image2.save(os.path.join(app.config['UPLOAD_FOLDER'], image2.filename))
            return "success"
        else:
            return 'File not allowed'
    return False

if __name__ == "__main__":
    app.run(debug=True)