# Microservices Security and Zero Trust Architecture

## 1. Zero Trust Principles in Cloud-Native Systems
Traditional perimeter-based security (castle-and-moat) assumes that all internal network traffic is trustworthy. Zero Trust discards this assumption under the maxim "never trust, always verify". Every inter-service request must be authenticated, authorized, and encrypted, regardless of network origin.

## 2. Mutual TLS (mTLS) Transport Security
In a microservices mesh, mutual TLS (mTLS) provides end-to-end cryptographic transport security.
- Unlike one-way TLS where only the server presents a certificate, mTLS requires both the client and the server to present valid X.509 certificates issued by a trusted Certificate Authority (CA).
- mTLS prevents Man-in-the-Middle (MITM) attacks and guarantees mutual authentication between microservice instances.
- Service meshes such as Istio and Linkerd automate certificate rotation and mTLS handshakes via sidecar proxies.

## 3. Identity and Authorization with OAuth2 and JWT
While mTLS authenticates service identity, user identity and resource authorization are managed using OAuth2 and JSON Web Tokens (JWT).
- An API Gateway intercepts incoming external traffic, validates the bearer JWT issued by an OAuth2 Identity Provider (IdP), and extracts claims (scopes, user ID, tenant ID).
- Downstream microservices evaluate Role-Based Access Control (RBAC) policies against the token claims to prevent Insecure Direct Object References (IDOR).

## 4. API Gateway Rate Limiting
To defend against Distributed Denial of Service (DDoS) and API abuse, modern API Gateways employ the Token Bucket algorithm.
- Each client key is assigned a bucket that refills with tokens at a steady rate.
- Requests consume tokens; when the bucket is empty, subsequent requests receive HTTP 429 Too Many Requests until tokens refill.
