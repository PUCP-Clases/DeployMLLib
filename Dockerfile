FROM python:3.7-stretch
RUN apt-get update
RUN apt-get install default-jdk -y
EXPOSE 8080
COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt
ENV PYSPARK_PYTHON=python3
COPY . /app
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
