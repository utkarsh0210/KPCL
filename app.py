# from flask import Flask, render_template, request
# from similarity import start_search

# app = Flask(__name__)

# # Dummy function simulating your processing logic
# def process_description(text):
#     return f"Best match for: '{text}' is → 'ALUMINIUM ROUND BAR'"

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     result = {}
#     ret_time = 0.0
#     record_count = 0
#     if request.method == 'POST':
#         user_input = request.form['description']
#         result, ret_time, record_count = start_search(user_input)
#     return render_template('index.html', result=result, retrieval_time=ret_time, record_count=record_count)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template
import os
from similarity import start_search
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Dummy function simulating your processing logic
def process_description(text):
    return f"Best match for: '{text}' is → 'ALUMINIUM ROUND BAR'"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = []
    ret_time = 0
    record_count = 0
    limit = None

    if request.method == 'POST':
        # Handle description search
        user_input = request.form.get('description')
        limit = request.form.get('limit', type=int)
        
        # Handle file upload
        uploaded_file = request.files.get('file')
        if uploaded_file and uploaded_file.filename:
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(file_path)            
            df = pd.read_csv(file_path, sep=',', encoding='latin-1')
            result, ret_time, record_count = start_search(user_input,df,limit)

            if limit > record_count:
                limit = record_count

    return render_template('index.html', result=result, retrieval_time=ret_time,limit = limit, record_count=record_count)