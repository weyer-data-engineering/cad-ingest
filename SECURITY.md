# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately, not in a public issue: on the repository's **Security** tab choose
**Report a vulnerability** (GitHub private vulnerability reporting). Include the version or image digest, the file
or request that triggers the problem, and what happens. You will get an answer within a few working days, and a fix
or a mitigation is released as soon as it is ready. Reporters are credited in the release notes unless they prefer
not to be.

## Supported versions

Security fixes go into the latest release (`ghcr.io/weyer-data-engineering/cad-ingest:latest` and its `vX.Y.Z` tag)
and into `main`. Please update before reporting.

## How cad-ingest handles untrusted input

CAD files, photos and scans are untrusted input to large C and C++ parsers (LibreDWG, OpenCascade, IfcOpenShell,
OpenCV, libheif). cad-ingest therefore:

- runs every job in its own work-horse process (own process group) with a hard timeout and an address-space limit,
  so a crash or a runaway file ends that job only;
- streams downloads into a private temporary directory with the size limit enforced while streaming, and removes it
  afterwards in every case;
- never uses the file name of a job as a path;
- never logs URLs, which usually carry signatures, only their host;
- runs as a non-root user (10001) and works with a read-only root filesystem when `/tmp` is writable.

## Deployment advice

- **Do not expose the job API to untrusted clients.** A job tells the worker which URLs to download from, upload to
  and call back. Anyone who can submit jobs can make the worker send HTTP requests to hosts in its network. Keep the
  API internal and set `API_TOKEN`.
- Give the worker container a memory limit (each conversion may reserve 85 % of it) and a read-only root filesystem
  with a writable `/tmp`.
- Use presigned URLs with short lifetimes for `source_url`, `preview_url` and `result_url`.
