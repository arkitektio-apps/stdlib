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
COPY poetry.lock /
RUN poetry install


# Set working directory
COPY init.py /tmp
WORKDIR /tmp
RUN python init.py

# Copy app
COPY . /app
WORKDIR /app

