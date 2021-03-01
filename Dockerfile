FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

WORKDIR /var/app

COPY app/ .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "simpleapp:app", "--host", "0.0.0.0", "--port", "8000"]
