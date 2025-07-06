from flask import Flask
from flask_cors import CORS
from routes.predict import predict_route
import subprocess

app = Flask(__name__)
CORS(app)
app.register_blueprint(predict_route)

if __name__ == '__main__':
    print("Iniciando aplicação...")
    app.run(debug=True)
