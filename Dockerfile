FROM amd64/python:3.10-slim as builder

RUN pip install poetry==1.4.0
WORKDIR /vin_decoder
COPY ./app ./app
COPY ./config ./config
COPY ./vin_decoder ./vin_decoder
COPY ./stores/model ./stores/model
COPY pyproject.toml poetry.lock poetry.toml ./

RUN poetry install --without dev,test,style

# Reducing image size from 1.7 gb to 1.0 gb
FROM amd64/python:3.10-slim as base

COPY --from=builder /vin_decoder /vin_decoder

WORKDIR /vin_decoder
ENV PATH="/vin_decoder/.venv/bin:$PATH"
ENTRYPOINT ["gunicorn", "-c", "app/gunicorn.py", "-k", "uvicorn.workers.UvicornWorker", "app.catboost_app:app"]
