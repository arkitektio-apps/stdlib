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
WORKDIR /app

# Run app on init to ensure dependencies are installed
#RUN python app.py 
