FROM tiangolo/uwsgi-nginx-flask:python3.7

COPY ./app /app
RUN pip install scikit-learn pandas flask gunicorn tensorflow numpy Jinja2 catboost
