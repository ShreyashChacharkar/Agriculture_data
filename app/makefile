# Makefile

# Build Docker image
build:
	docker build -t streamlit-app .

# Run Docker container
run:
	docker run -p 8501:8501 streamlit-app

# Remove Docker container and image
clean:
	docker stop $$(docker ps -q --filter ancestor=streamlit-app)
	docker rmi streamlit-app
