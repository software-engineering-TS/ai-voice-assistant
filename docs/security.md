# Security and deployment boundaries

## Default deployment

The application defaults to `HOST=127.0.0.1` and is intended for local development. Keep that setting for personal use.

## Optional shared API token

Set a long, randomly generated `API_ACCESS_TOKEN` in `backend/.env`. Set the same value in `streamlit/.env`. The FastAPI middleware then requires:

```http
Authorization: Bearer <API_ACCESS_TOKEN>
```

The token applies to API endpoints and generated audio files. The health endpoint, root metadata endpoint and API documentation remain public so that local tools can inspect service availability and schema.

This is a shared-token gate. It is **not** multi-user authentication, identity verification, account ownership, role-based authorization, per-user data isolation or rate limiting. Do not use it as a substitute for those controls in a public product.

## Remote access

Before setting `HOST` to a non-loopback interface:

1. Configure a strong random `API_ACCESS_TOKEN` on both backend and Streamlit.
2. Place the services behind an HTTPS reverse proxy.
3. Restrict inbound network access at the host/firewall layer.
4. Configure `ALLOWED_ORIGINS` to exact trusted browser origins; avoid wildcard origins.
5. Keep generated audio and upload directories outside public web roots except through the protected API.
6. Define file retention and cleanup, backups, log access, monitoring and rate limits.
7. Add a proper identity and authorization layer before supporting multiple users.

CORS is not authentication. It does not stop command-line clients or direct HTTP requests.

## File handling

- Session IDs are restricted to alphanumeric characters, hyphens and underscores.
- Audio input filenames are never used as storage paths; uploads get random server-side names.
- Audio uploads have a configurable size limit and are removed after processing.
- Generated audio uses random names to avoid session collisions and path traversal.
- Document ingestion uses extension validation and a separate upload-size limit.

Uploaded content can contain personal or confidential information. Do not store it longer than necessary, and do not upload sensitive information unless you have permission and appropriate controls.

## Known gaps

The shared token does not provide per-user ownership checks for conversations or documents. The current application should therefore be treated as a trusted single-user/local tool until proper authentication, authorization and rate limiting are implemented.