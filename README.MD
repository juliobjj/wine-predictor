# Wine Quality Predictor 🍷

Este projeto faz predições de qualidade de vinho com base em atributos químicos, utilizando modelos de machine learning treinados com Scikit-Learn.

## Tecnologias
- Python + Flask
- Scikit-Learn
- Pytest
- Front-end com HTML/JS simples

# Como Executar Testes

python -m pytest -s tests/test_model_quality.py

# Inicie o back-end:

```bash
# cria ambiente virtual
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows


# instala dependências
pip install -r requirements.txt

# roda a aplicação
python app.py
