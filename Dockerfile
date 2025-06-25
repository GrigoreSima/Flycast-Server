# Base image with Java and Python
FROM openjdk:21-jdk-slim

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    apt-get clean

# Set work directory
WORKDIR /app

# Copy your Spring Boot JAR
COPY ./server/Flycast-Server-0.0.1-SNAPSHOT.jar ./server/app.jar

# Copy your Python scripts
COPY predictor/ ./predictor/

# Copy Python dependencies
COPY requirements.txt .

# Create Python virtual environment
RUN python3 -m venv /opt/venv

# Install Python dependencies in the virtual environment
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Add virtualenv Python to PATH
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app/server

# Run your Spring Boot app
CMD ["java", "-jar", "app.jar"]
