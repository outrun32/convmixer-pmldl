# Use an official Python runtime as a parent image
FROM python:3.9-slim

WORKDIR /app

# Copy the necessary files into the container
COPY code/ /app/code/
COPY models/ /app/models/

ENV PYTHONPATH=/app

# Install any needed packages specified in requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install streamlit torch torchvision pillow timm

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "code/deployment/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
