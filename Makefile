build-dev:
	docker-compose -f docker-compose-dev.yaml up --build -d
build:
	docker-compose up --build -d