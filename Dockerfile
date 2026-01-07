FROM python:3.9-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Copier et installer les dépendances ENCORE en root
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# 3. Copier le code applicatif
COPY app/ ./app/
COPY model/churn_model.pkl ./model/

# 4. Créer l'user non-root et lui donner les droits
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# 5. IMPORTANT : Donner l'accès EXÉCUTION à l'user
RUN chmod +x /app/app/main.py 2>/dev/null || true

# 6. Passer à l'user non-root
USER appuser

# 7. Vérifier que l'user peut lire les fichiers
RUN python -c "import os; print(f'UID: {os.getuid()}, GID: {os.getgid()}')" && \
    ls -la /app/model/ 2>/dev/null | head -3

ENV PORT=8000 \
    MODEL_PATH=/app/model/churn_model.pkl

EXPOSE 8000

# 8. Healthcheck avec curl (installé plus tôt)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]