.PHONY: build attach scale down

GPU ?= 0

attach: build
	docker compose up -d app$(GPU)
	docker attach quash-vlmaps-app$(GPU)-1

build:
	docker compose build app$(GPU)

down:
	docker compose down

scale:
ifndef N
	$(error Usage: make scale GPU=X N=N)
endif
	docker compose up -d app$(GPU) --scale app$(GPU)=$(N)