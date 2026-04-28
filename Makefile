.PHONY: build up down bash log

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

bash:
	docker exec -it rviser bash

log:
	docker compose logs -f