FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY src ./src

RUN addgroup --system appgroup \
	&& adduser --system --ingroup appgroup --home /home/appuser appuser \
	&& mkdir -p /home/appuser \
	&& chown -R appuser:appgroup /app /home/appuser

ENV HOME=/home/appuser
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
