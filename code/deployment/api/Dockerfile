# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files into the container
COPY code/ /app/code/
COPY models/ /app/models/
# Install any needed packages specified in requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install fastapi uvicorn torch torchvision pillow timm python-multipart

ENV PYTHONPATH=/app

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "code.deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
