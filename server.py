# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

#업로드 html 렌더링
@app.route('/')
def render_file():
    return render_template('upload.html')

#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    if request.method =='POST':
        f = request.file['file']
        #저장할 경로 + 파일명
        f.save(secure_filename(f.filename))
        return 'uploads 디렉토리 --> 파일 업로드 성공!'


#def index():

 #       return render_template('index.html')
if __name__ == '__main__':

   app.run(debug = True)
