# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt .

# Create and activate a virtual environment
RUN python -m venv venv
RUN . venv/bin/activate

# Upgrade pip and install dependencies
RUN pip install -U pip
RUN pip install -r requirements.txt

# Expose port you want your app on
EXPOSE 8080

# Copy the rest of the application files to the container
COPY . .

# Run Streamlit
CMD ["streamlit", "run", "--server.port", "8080", "app.py"]
