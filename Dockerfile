FROM python:3.10

# Install dependencies
RUN pip install --upgrade pip
# Install dependencies
RUN pip install poetry rich

# Configure poetry
RUN poetry config virtualenvs.create false 
ENV PYTHONUNBUFFERED=1


# Copy dependencies
COPY pyproject.toml /
RUN poetry install

# Set working directory
COPY init.py /tmp
WORKDIR /tmp
RUN python init.py

RUN pip install "scikit-learn"
RUN pip install "arkitekt[all]==0.5.58"



# Copy app
COPY . /app
WORKDIR /app

