from detect import *
from flask import Flask, redirect, url_for, request


import requests, json
import os, shutil

app = Flask(__name__)

@app.route('/')
def home():
	return 'Welcome to the Deep Learning End Point created by Jitesh Motwani && Karan Dev'


@app.route('/deeplearning/api/object_detection',methods = ['POST', 'GET'])
def object_detection():
	folder = "data/api_sample"
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
	try:
		if os.path.isfile(file_path):
			os.unlink(file_path)
		#elif os.path.isdir(file_path): shutil.rmtree(file_path)
	except Exception as e:
		print(e)
	if request.method == 'GET':
		url = request.args.get('url')
		f = open(folder+"/nemo_sample.jpeg",'wb')
		f.write(requests.get(url).content)
		f.close()
		output = detection(folder)
	
	return json.dumps(output)

if __name__ == '__main__':
	app.run(debug = True)
