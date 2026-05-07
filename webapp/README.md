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

## Deploy on Vercel

Vercel is recommended for the web/PWA layer, not for running the 7B LoRA directly.
Deploy the model on a GPU backend, then let Vercel proxy requests to it.

1. Import the GitHub repository in Vercel.
2. Set the project root directory to `webapp`.
3. Use framework preset `Other`.
4. Add environment variables:

```text
LATIN_BACKEND_URL=https://your-gpu-backend.example.com/generate
LATIN_BACKEND_TOKEN=optional-shared-secret
```

On Vercel, the frontend uses `/api/generate` automatically. That serverless function forwards the browser request to `LATIN_BACKEND_URL`, keeping the real GPU endpoint configurable.

For local static testing with `python -m http.server`, configure the backend URL manually in the app settings, for example `http://localhost:8000/generate`.

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
