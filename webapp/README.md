# Agente Tutor Latinista Web App

Minimal PWA frontend for the Latin Academic LoRA tutor.

## Run Locally

This is a static app. From the project root:

```bash
python -m http.server 5173 -d webapp
```

Open:

```text
http://localhost:5173
```

## Backend Endpoint

The app expects a JSON endpoint:

```http
POST /generate
Content-Type: application/json

{"prompt": "Declina en todos los casos: servus (m, 2da)"}
```

Response:

```json
{"answer": "Nom. servus | Gen. servi ..."}
```

Use `server/latin_tutor_api.py` as the reference backend.

## Android Path

Recommended options:

1. PWA install from Chrome on Android.
2. Wrap this static app with Capacitor.
3. Use a hosted backend for model inference. Do not run the 7B model directly on a phone.

Capacitor sketch:

```bash
npm create @capacitor/app
npm install @capacitor/android
npx cap add android
```

Then copy the static files from `webapp/` into the Capacitor web directory.
