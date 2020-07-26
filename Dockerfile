FROM python:3.7
EXPOSE 8503
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run my_app.py --server.port 8503