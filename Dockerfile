FROM tiangolo/uwsgi-nginx-flask:python3.7-2023-07-17

COPY ./app /app
RUN  pip install --no-cache-dir scikit-learn==0.21.3 pandas==0.25.1 Flask==1.1.1 gunicorn numpy==1.17.2 Jinja2==2.10.3 catboost==0.21 tensorflow==2.1.0 sklearn-pandas==1.8.0 pip install markupsafe==2.0.1 itsdangerous==2.0.1 werkzeug==2.0.3 protobuf==3.20.*
