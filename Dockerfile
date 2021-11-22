FROM python:3.9-slim

RUN mkdir /python /python/output
WORKDIR /python
COPY requirements.txt setup.py ./
COPY solution solution
COPY tests tests
RUN pip3 install -r requirements.txt \
    && pip install -e . \
    && pytest tests

CMD ["python", "solution/train_agent.py"]



