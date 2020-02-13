FROM tiangolo/uwsgi-nginx-flask:python3.7

COPY ./app /app
RUN  pip install --no-cache-dir scikit-learn==0.21.3 pandas==0.25.1 Flask==1.1.1 gunicorn numpy==1.17.2 Jinja2==2.10.3 catboost==0.21
