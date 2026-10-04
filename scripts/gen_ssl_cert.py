"""
AURA SSL Certificate Generator
Generates a self-signed TLS certificate for serving AURA over HTTPS.
Allows mobile devices to access the camera via getUserMedia without Chrome flags.

Usage:
    python scripts/gen_ssl_cert.py
    python scripts/gen_ssl_cert.py --out-dir certs --days 3650
"""

import os
import sys
import argparse
import ipaddress
import datetime

def gen_cert(out_dir: str = ".", days: int = 3650):
    """Generate a self-signed cert + key using cryptography library."""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import socket
    except ImportError:
        print("[AURA] 'cryptography' package not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cryptography"])
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import socket

    os.makedirs(out_dir, exist_ok=True)
    cert_path = os.path.join(out_dir, "cert.pem")
    key_path  = os.path.join(out_dir, "key.pem")

    # Collect all local IPs
    local_ips = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            addr = info[4][0]
            try:
                local_ips.add(ipaddress.ip_address(addr))
            except ValueError:
                pass
    except Exception:
        pass
    local_ips.add(ipaddress.ip_address("127.0.0.1"))

    # Generate RSA key
    print("[AURA] Generating RSA 2048 private key...")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Build certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "IN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AURA Local"),
        x509.NameAttribute(NameOID.COMMON_NAME, "aura.local"),
    ])

    san_list = [x509.DNSName("localhost"), x509.DNSName("aura.local")]
    for ip in local_ips:
        san_list.append(x509.IPAddress(ip))

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))

    print(f"[AURA] Certificate: {cert_path}")
    print(f"[AURA] Private key: {key_path}")
    print(f"[AURA] Valid {days} days. SANs: {[str(i) for i in local_ips]}")
    return cert_path, key_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AURA self-signed SSL cert generator")
    parser.add_argument("--out-dir", default="certs", help="Output directory. Default: certs/")
    parser.add_argument("--days", type=int, default=3650, help="Validity in days. Default: 3650")
    args = parser.parse_args()
    gen_cert(args.out_dir, args.days)
