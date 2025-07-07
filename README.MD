# Criar ambiente virtual

cd backend
python3 -m venv venv
source venv/bin/activate # Linux/macOS
venv\Scripts\activate # Windows

# Como Instalar Dependências

pip install -r ../requirements.txt

# Como Executar Testes

python -m pytest -s tests/test_model_quality.py

# Inicie o back-end:

python app.py

Abra o arquivo ../frontend/index.html no navegador.
