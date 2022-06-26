from click import echo
import flask
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from tensorflow.keras import models
import numpy as np
from tensorflow.keras.preprocessing import image
# from flask_mysqldb import MySQL
import base64
from PIL import Image
import io

flask_app = flask.Flask(__name__)

# flask_app.config['MYSQL_HOST'] = 'localhost'
# flask_app.config['MYSQL_USER'] = 'root'
# flask_app.config['MYSQL_PASSWORD'] = ''
# flask_app.config['MYSQL_DB'] = 'proyek akhir'

# mysql = MySQL(flask_app)

app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "Skin Cancer Detection", 
		  description = "Classify skin cancer using a trained CNN model")

name_space = app.namespace('prediction', description='Classification APIs')

model = app.model('Classification parameters', 
				  {'file': fields.String(required = True, 
				  							   description="Image to classify", 
    					  				 	   help="Image cannot be blank")})

loaded_model = models.load_model('18052022-experimentxxx-withbatchnorm-add.h5')

@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response

	@app.expect(model)		
	def post(self):
		imagefile = request.files['file']
		filename = werkzeug.utils.secure_filename(imagefile.filename)
		print("\nReceived image file name : " + filename)
		print('Predicting.....')
		img = Image.open(imagefile, 'r')
		img = img.convert('RGB')
		buf = io.BytesIO()
		img.save(buf, format='JPEG')
		img_byte = buf.getvalue()
		file = base64.b64encode(img_byte)
		img = img.resize((100, 100), Image.NEAREST)
		img_array = image.img_to_array(img).astype('float32')/255
		img_array = np.expand_dims(img_array, axis=0)

		predictions = loaded_model.predict(img_array)
		class_names = ['benign', 'malignant']
		score = predictions[0]
		print('Probability -->', class_names[0], ':', round(100 * score[0], 1),'%,', class_names[1], ':', round(100 * score[1], 1),'%')
		predicted_label = str("Gambar ini kemungkinan besar tergolong {} dengan probabilitas {:.1f}%."
    		.format(class_names[np.argmax(score)], 100 * np.max(score)))
		print('Predicted class :', class_names[np.argmax(score)])
		class1 = None
		score1 = None
		class2 = None
		score2 = None
		if(class_names[np.argmax(score)]  == 'benign' and class_names[np.argmin(score)] == 'malignant'):
			class1 = class_names[np.argmax(score)]
			score1 = 100 * np.max(score)
			class2 = class_names[np.argmin(score)]
			score2 = 100 * np.min(score)
		elif (class_names[np.argmax(score)] == 'malignant' and class_names[np.argmin(score)] == 'benign'):
			class1 = class_names[np.argmin(score)]
			score1 = 100 * np.min(score)
			class2 = class_names[np.argmax(score)]
			score2 = 100 * np.max(score)
		result = [
			{
				"class" : class1, 
				"score" : score1
			}, 
			{
				"class" : class2, 
				"score" : score2
			},
			{
				"predicted_label" : predicted_label
			}
		]
		response = jsonify({
			"statusCode": 200,
			"status": "Classification result",
			"result": result
		})

		# try:
		# 	cursor = mysql.connection.cursor()
		# 	cursor.execute(''' INSERT INTO data_prediksi(nama_file, gambar, klasifikasi) VALUES(%s,%s,%s)''', (filename, file, class_names[np.argmax(score)]))
		# 	mysql.connection.commit()
		# 	cursor.close()
			
		# except Exception as e:
		# 	print(str(e))
		# 	return jsonify("error")

		response.headers.add("Access-Control-Allow-Origin", "*")
		return response

if __name__ == '__main__':
	flask_app.run(threaded=True, port=5000)