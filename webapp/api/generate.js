export default async function handler(request, response) {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return response.status(405).json({ error: "Method not allowed" });
  }

  const backendUrl = process.env.LATIN_BACKEND_URL;
  if (!backendUrl) {
    return response.status(500).json({
      error: "LATIN_BACKEND_URL is not configured",
    });
  }

  try {
    const upstream = await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.LATIN_BACKEND_TOKEN
          ? { Authorization: `Bearer ${process.env.LATIN_BACKEND_TOKEN}` }
          : {}),
      },
      body: JSON.stringify(request.body),
    });

    const text = await upstream.text();
    response.status(upstream.status);
    response.setHeader("Content-Type", upstream.headers.get("content-type") || "application/json");
    return response.send(text);
  } catch (error) {
    return response.status(502).json({
      error: "Backend request failed",
      detail: error.message,
    });
  }
}
