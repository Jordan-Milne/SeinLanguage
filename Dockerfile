FROM tiangolo/uwsgi-nginx-flask:python3.7

COPY ./app /app
RUN  pip install --no-cache-dir scikit-learn pandas flask gunicorn numpy Jinja2 catboost tensorflow
