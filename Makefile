IMAGE_NAME=api4model2cloud
EXT_PORT=8000
INT_PORT=8000

build:
	docker build -t $(IMAGE_NAME) .

build-nc:
	docker build --no-cache -t $(IMAGE_NAME) .

run:
	docker run --rm -t -i -p $(EXT_PORT):$(INT_PORT) --name="$(IMAGE_NAME)" $(IMAGE_NAME)
