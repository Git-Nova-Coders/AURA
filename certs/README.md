# certs/

This folder holds the auto-generated TLS certificate for serving AURA over **HTTPS**.
HTTPS is required so mobile browsers allow camera access via `getUserMedia`.

## Files

| File | Description | Git |
|------|-------------|-----|
| `cert.pem` | Public certificate (self-signed) | ❌ gitignored — machine-specific |
| `key.pem`  | **Private key — NEVER commit this** | ❌ gitignored |

## Generation

Certificates are **automatically generated on first run** of `run_dashboard.py`.

To regenerate manually:
```bash
python scripts/gen_ssl_cert.py --out-dir certs
```

The certificate is valid for 10 years and covers all local network IPs
(e.g. `10.x.x.x`, `192.168.x.x`, `127.0.0.1`) so any phone on the same
Wi-Fi can open `https://<laptop-ip>:8420/remote-camera`.

## Bypassing the browser warning

Because this is self-signed (not issued by a public CA), browsers will show
a "Your connection is not private" warning the **first time**:

- **Laptop Chrome**: Click *Advanced* → *Proceed to localhost (unsafe)*
- **Phone Chrome**: Tap *Advanced* → *Proceed to \<IP\> (unsafe)*
- **Phone Safari**: Tap *Show Details* → *visit this website*

This only needs to be done once per browser.
