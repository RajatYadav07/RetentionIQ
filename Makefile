.PHONY: setup data train api dashboard docker-up docker-down test all

setup:
	pip install -r requirements.txt

data:
	python scripts/generate_data.py

train:
	python scripts/train_model.py

api:
	uvicorn app.api.main:app --reload

dashboard:
	streamlit run app/dashboard/main.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

test:
	pytest tests/

all: data train
