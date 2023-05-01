## Introduction

### Installation

We need to install `python 3.10.9` and create virtualenv.

The recommended way to do so is to use [pyenv](https://github.com/pyenv/pyenv) to manage python versions and [poetry](https://python-poetry.org/docs/) to install and manage python dependencies as well as build virtual environments and package your code.

__N.B.__ it's better to install `poetry=1.4.0` by running
```bash
curl -sSL https://install.python-poetry.org | python3 - --version 1.4.0
```
and then adding it to your `PATH`.

Once the aforesaid tools are installed, we can run the following commands to install the necessary python version:

```bash
make python
```
Now let's build a virtual environment to isolate dependencies from the packages in the base environment:

```bash
make env
```
That's it!

### ML pipeline
```bash
python vin_decoder/main.py extract-data
python vin_decoder/main.py validate-data
python vin_decoder/main.py preprocess-labels
python vin_decoder/main.py optimize-train-model --experiment-name="baseline_vin_decoder"
python vin_decoder/main.py predict-vehicle-model --vin="WBAFW11000DUXXXXX"
```

### MLflow
```bash
mlflow server -h 0.0.0.0 -p 8000 --backend-store-uri $PWD/stores/model
```

### REST API

```bash
uvicorn app.catboost_app:app --host 0.0.0.0 --port 8000 --reload --reload-dir vin_decoder --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.catboost_app:app  # prod
```

### Docker
```bash
docker build -t vin_decoder:latest -f Dockerfile .
docker run -p 8000:8000 --name vin_decoder vin_decoder:latest
```

#### Curl

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "vins": [
    {
      "vin": "WAUZZZ8T39A0XXXXX"
    },
    {
      "vin": "WAUZZZ4G0BN0XXXXX"
    }
  ]
}'
```

#### Request URL

```
http://localhost:8000/predict
```

##### Response body
```bash
{
  "message": "OK",
  "method": "POST",
  "status-code": 200,
  "timestamp": "2023-04-30T12:37:13.079220",
  "url": "http://localhost:8000/predict",
  "data": {
    "vins": [
      "WAUZZZ8T39A0XXXXX",
      "WAUZZZ4G0BN0XXXXX"
    ],
    "predictions": [
      "A5",
      "A6"
    ]
  }
}
```

##### Response headers

```
content-length: 213
content-type: application/json
date: Sun,30 Apr 2023 12:37:12 GMT
server: uvicorn
```

#### Swagger UI
```bash
http://localhost:8000/docs
```
