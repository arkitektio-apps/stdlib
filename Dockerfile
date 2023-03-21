FROM python:3.11

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade wheel

# Copy requirements.txt
COPY requirements.txt /tmp/requirements.txt

# Install requirements
RUN pip install -r /tmp/requirements.txt

# Copy source code
COPY . /app

# Set working directory
WORKDIR /app

# Run app on init to ensure dependencies are installed
RUN python app.py 
