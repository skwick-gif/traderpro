# Docker & Compose (Dev / Prod)

## דרישות
- Docker Desktop/Engine מותקן.
- TWS/IB Gateway פעיל עם Enable API (7497 Paper / 7496 Live).

## קבצים
- backend.Dockerfile.dev / frontend.Dockerfile.dev — פיתוח (hot-reload + bind mounts)
- backend.Dockerfile / frontend.Dockerfile — פרודקשן
- docker-compose.dev.yml / docker-compose.prod.yml
- .env.example — למלא מפתחות ספקים ופרטי IBKR

## הפעלה בפיתוח
```bash
cp .env.example .env   # מלא מפתחות
docker compose -f docker-compose.dev.yml up --build
```
- Backend: http://localhost:8000
- Frontend: http://localhost:3000

## הפעלה בפרודקשן
```bash
docker compose -f docker-compose.prod.yml up --build -d
```
- Frontend: http://localhost:8080
- Backend: http://localhost:8000

## לוגים ותקלות
```bash
docker compose -f docker-compose.dev.yml logs -f backend
docker compose -f docker-compose.dev.yml logs -f frontend
```
