**本地可直接落地的完整 MLOps + DVC 实操版**

---

# 0. 最终你将获得

- MLflow（实验追踪 + 模型注册）
- MinIO（Artifact + DVC 数据远端）
- PostgreSQL（MLflow 元数据）
- DVC（数据版本管理 + pipeline）
- FastAPI（模型服务）
- Makefile（一键执行）

---

# 1. 项目结构（完整）

```bash
mlops-local/
├── docker-compose.yaml
├── Dockerfile.mlflow
├── requirements.txt
├── Makefile
├── dvc.yaml
├── params.yaml
├── train.py
├── app/
│   └── serving.py
├── scripts/
│   ├── prepare_data.py
│   └── init_minio_buckets.sh
├── data/
│   └── raw/
│       └── iris.csv
└── models/
```

---

# 2. 基础设施：Docker Compose

## `Dockerfile.mlflow`
```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow psycopg2-binary boto3
EXPOSE 5000
```

## `docker-compose.yaml`
```yaml
services:
  db:
    image: postgres:16
    container_name: mlflow_db
    restart: always
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow_password
      POSTGRES_DB: mlflow_db
    volumes:
      - ./postgres_data:/var/lib/postgresql/data
    networks:
      - mlops-network

  minio:
    image: minio/minio
    container_name: mlops_minio
    restart: always
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: password123
    volumes:
      - ./minio_data:/data
    command: server /data --console-address ":9001"
    networks:
      - mlops-network

  create_buckets:
    image: minio/mc
    depends_on:
      - minio
    networks:
      - mlops-network
    entrypoint: >
      /bin/sh -c "
      until (/usr/bin/mc alias set local http://minio:9000 admin password123); do echo waiting_minio; sleep 1; done;
      /usr/bin/mc mb --ignore-existing local/mlflow;
      /usr/bin/mc mb --ignore-existing local/dvc;
      exit 0;
      "

  mlflow:
    build:
      context: .
      dockerfile: Dockerfile.mlflow
    container_name: mlflow_server
    restart: always
    depends_on:
      - db
      - minio
      - create_buckets
    ports:
      - "5000:5000"
    environment:
      AWS_ACCESS_KEY_ID: admin
      AWS_SECRET_ACCESS_KEY: password123
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow_password@db:5432/mlflow_db
      --default-artifact-root s3://mlflow/
      --host 0.0.0.0
      --port 5000
    networks:
      - mlops-network

networks:
  mlops-network:
    driver: bridge
```

启动：
```bash
docker compose up -d --build
```

---

# 3. Python 依赖

## `requirements.txt`
```txt
mlflow
dvc[s3]
boto3
pandas
scikit-learn
pyyaml
fastapi
uvicorn
```

安装：
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

# 4. 准备数据脚本

## `scripts/prepare_data.py`
```python
from sklearn.datasets import load_iris
import pandas as pd
from pathlib import Path

Path("data/raw").mkdir(parents=True, exist_ok=True)

X, y = load_iris(return_X_y=True, as_frame=True)
df = X.copy()
df["target"] = y
df.to_csv("data/raw/iris.csv", index=False)
print("saved: data/raw/iris.csv")
```

运行：
```bash
python scripts/prepare_data.py
```

---

# 5. DVC 初始化与远端（MinIO）

```bash
git init
dvc init
git add .dvc .dvcignore
git commit -m "init dvc"

dvc remote add -d minio_remote s3://dvc/
dvc remote modify minio_remote endpointurl http://127.0.0.1:9000
dvc remote modify minio_remote access_key_id admin
dvc remote modify minio_remote secret_access_key password123
dvc remote modify minio_remote use_ssl false
```

纳管数据：
```bash
dvc add data/raw/iris.csv
git add data/raw/iris.csv.dvc .gitignore
git commit -m "track dataset by dvc"
dvc push
```

---

# 6. 训练参数

## `params.yaml`
```yaml
train:
  test_size: 0.2
  random_state: 42
  n_estimators: 100
  max_depth: 4
```

---

# 7. 训练脚本（MLflow + DVC 元信息）

## `train.py`
```python
import os
import json
import hashlib
import subprocess
from pathlib import Path

import yaml
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def run_cmd(cmd: str):
    return subprocess.check_output(cmd, shell=True, text=True).strip()


def file_md5(path: str):
    m = hashlib.md5()
    with open(path, "rb") as f:
        m.update(f.read())
    return m.hexdigest()


def get_git_commit():
    try:
        return run_cmd("git rev-parse HEAD")
    except Exception:
        return "N/A"


def get_dvc_meta(path="data/raw/iris.csv.dvc"):
    # 读取.dvc文件内容用于记录
    p = Path(path)
    if not p.exists():
        return {"dvc_file": path, "dvc_meta_md5": "N/A", "dvc_out_md5": "N/A"}
    d = yaml.safe_load(p.read_text())
    out_md5 = d.get("outs", [{}])[0].get("md5", "N/A")
    return {
        "dvc_file": path,
        "dvc_meta_md5": file_md5(path),
        "dvc_out_md5": out_md5
    }


def main():
    # 本地访问 docker 服务
    MLFLOW_URI = "http://127.0.0.1:5000"
    S3_ENDPOINT = "http://127.0.0.1:9000"

    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = S3_ENDPOINT

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("iris-exp")

    params = yaml.safe_load(Path("params.yaml").read_text())["train"]

    df = pd.read_csv("data/raw/iris.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"],
        random_state=params["random_state"]
    )

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    Path("models").mkdir(parents=True, exist_ok=True)

    git_commit = get_git_commit()
    dvc_meta = get_dvc_meta("data/raw/iris.csv.dvc")

    with mlflow.start_run() as run:
        # 参数与指标
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        # 可追溯信息
        mlflow.log_param("git_commit", git_commit)
        for k, v in dvc_meta.items():
            mlflow.log_param(k, v)

        # 保存本地结果（供dvc pipeline追踪）
        metrics = {"accuracy": acc, "run_id": run.info.run_id}
        Path("models/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        # 记录模型到MLflow并注册
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="iris_rf"
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
```

---

# 8. DVC Pipeline

## `dvc.yaml`
```yaml
stages:
  train:
    cmd: python train.py
    deps:
      - train.py
      - data/raw/iris.csv
      - params.yaml
    outs:
      - models/metrics.json
```

执行：
```bash
dvc repro
```

提交：
```bash
git add dvc.yaml dvc.lock params.yaml train.py
git commit -m "add training pipeline with mlflow+dvc traceability"
```

---

# 9. 推理服务

## `app/serving.py`
```python
import os
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc

os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.pyfunc.load_model("models:/iris_rf/latest")

app = FastAPI(title="Iris Inference Service")

class PredictRequest(BaseModel):
    data: List[List[float]]

@app.post("/predict")
def predict(req: PredictRequest):
    pred = model.predict(req.data).tolist()
    return {"predictions": pred}
```

运行：
```bash
uvicorn app.serving:app --host 0.0.0.0 --port 8000
```

测试：
```bash
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{"data": [[5.1,3.5,1.4,0.2],[6.2,3.4,5.4,2.3]]}'
```

---

# 10. Makefile（一键化）

## `Makefile`
```makefile
.PHONY: infra-up infra-down init-data dvc-setup dvc-track train repro serve

infra-up:
	docker compose up -d --build

infra-down:
	docker compose down

init-data:
	python scripts/prepare_data.py

dvc-setup:
	dvc init
	dvc remote add -d minio_remote s3://dvc/ || true
	dvc remote modify minio_remote endpointurl http://127.0.0.1:9000
	dvc remote modify minio_remote access_key_id admin
	dvc remote modify minio_remote secret_access_key password123
	dvc remote modify minio_remote use_ssl false

dvc-track:
	dvc add data/raw/iris.csv
	dvc push

train:
	python train.py

repro:
	dvc repro

serve:
	uvicorn app.serving:app --host 0.0.0.0 --port 8000
```

---

# 11. 本地运行顺序（照抄即可）

```bash
make infra-up
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make init-data
git init
make dvc-setup
make dvc-track

git add .
git commit -m "init local mlops with dvc"

make repro     # 或 make train
make serve
```
