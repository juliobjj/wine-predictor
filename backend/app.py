from flask import Flask
from flask_cors import CORS

from routes.predict import router
from database import Base, engine

# Inicialização
app = Flask(__name__)
CORS(app)

# Cria as tabelas no banco
Base.metadata.create_all(bind=engine)

# Registra o blueprint
app.register_blueprint(router)

if __name__ == '__main__':
    app.run(debug=True)
