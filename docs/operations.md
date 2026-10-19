# Operating cad-ingest

## Processes

One image, three components:

| Component | Start | What it does |
|---|---|---|
| `api` | default `CMD` (`uvicorn app.main:app`, port 8080) | HTTP API. Stateless: all state is in Redis, any number of replicas |
| `worker` | `docker run <image> worker` (command `worker`, i.e. `python -m app.worker`) | an RQ worker pool: `MAX_CONCURRENT_CONVERSIONS` workers, each forks one work-horse per job |
| Redis | `REDIS_URL` | queues and job state; any Redis 6 or later. A shared Redis is fine with a database index of its own, e.g. `redis://redis:6379/3` |

Health checks:

```bash
curl -f http://localhost:8080/health          # api
worker check                                  # worker: exit 0 if a worker of this host is registered in Redis
```

A complete setup is in [`docker-compose.yml`](../docker-compose.yml).

Queues: `cad-interactive` is served before `cad-bulk` (priority `interactive` / `bulk` per job), `cad-callbacks` before
both. All three live in the Redis database of `REDIS_URL`.

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis (api and worker) |
| `API_TOKEN` | not set | if set, `/jobs` requires `Authorization: Bearer <token>` (api) |
| `MAX_CONCURRENT_CONVERSIONS` | `2` | RQ workers per `worker` container, i.e. conversions in parallel (worker) |
| `CONVERSION_TIMEOUT_SECONDS` | `600` | hard limit per attempt (download, conversion and upload), RQ kills the work-horse |
| `MAX_FILE_SIZE_BYTES` | `524288000` | limit for the source (500 MB), enforced while streaming |
| `MEMORY_LIMIT_MB` | from the container | `RLIMIT_AS` of each work-horse. Not set: 85 % of the container memory limit (cgroup v2 `memory.max`, v1 `memory.limit_in_bytes`); without a container limit `8192`. It is not divided by `MAX_CONCURRENT_CONVERSIONS`, because it caps reserved address space, not real use |
| `MIN_FREE_DISK_MB` | `1024` | free space in the temp dir that must remain; below it a job is not failed but queued again later |
| `MAX_QUEUED_JOBS` | `10000` | waiting jobs (both priorities) before `POST /jobs` answers `503` (api) |
| `MAX_ATTEMPTS` | `3` | runs per job for failures that are not the file's fault, first run included |
| `RESULT_TTL_SECONDS` | `86400` | how long a finished job (markdown, error) can be read |
| `MAX_MARKDOWN_CHARS` | `2000000` | the markdown is truncated beyond this (with a warning) |
| `RETRY_DELAY_SECONDS` | `30` | first delay before a re-run (then x4: 30 s, 2 min) and before a callback retry (then x2) |
| `OCR_LANGUAGES` | `deu` | Tesseract languages for the texts on photos and scans: `eng` for English drawings, `deu+eng` for both (the image has `deu` and `eng`; add other `tesseract-ocr-*` packages in your own image) |

The environment is read when a job starts, so settings are per container: the api needs `REDIS_URL`, `API_TOKEN`,
`MAX_QUEUED_JOBS`; the worker everything else.

## Execution, failures and restarts

A job runs in an RQ work-horse (a forked process of its own, in its own process group) which

1. sets `oom_score_adj=1000` and `RLIMIT_AS` (`MEMORY_LIMIT_MB`), so the kernel's OOM killer picks a horse, never the
   worker, and a runaway allocation ends as `out_of_memory`;
2. streams `source_url` into a private temp dir (`cad-<job id>`, removed afterwards in every case) with the size limit
   enforced while streaming;
3. converts (`extract`, then `preview`), `PUT`s the preview to `preview_url`;
4. stores the result in Redis and queues the callback.

What happens on failure:

- **The file's fault** (`unsupported`, `unreadable`, `too_large`, `timeout`, `out_of_memory`, also a horse that crashed
  or was killed by the kernel while its worker lived): the job fails at once, `retryable: false`.
- **Not the file's fault** (download or preview upload fails with a connection error, `5xx`, `408`, `429`; a work-horse
  lost to a restart): RQ runs the job again, waiting `RETRY_DELAY_SECONDS` x 1, 4, 16 between runs, up to
  `MAX_ATTEMPTS` runs. While it waits the job is `queued`. After the last run it fails with `retryable: true`.
- **A job abandoned by a dying worker** (container killed or restarted, node lost): the worker's heartbeat stops. About
  70 seconds later RQ's registry clean-up (every 30 seconds, by any live worker) notices the abandoned job and
  re-queues it while it has runs left (the converter's exception handler keeps the job document in step); otherwise it
  fails with `internal`, `retryable: true`. So after a worker restart interrupted jobs are running again within about
  two minutes. A graceful stop (`SIGTERM`) lets running conversions finish first, so give the container a
  `stop_grace_period` that fits your files; what is still running afterwards is killed and handled as above.
- **Not enough free disk** (below `MIN_FREE_DISK_MB`, or the download would leave less): not a failure; the job is
  queued again after `RETRY_DELAY_SECONDS` without using an attempt, and again while the disk stays full.

What survives which restart:

| Restart of | Queued jobs | Running jobs | Finished results | Callbacks not yet delivered |
|---|---|---|---|---|
| api | kept | keep running | kept | keep being retried |
| worker | kept | re-run within about two minutes (or finish first on a graceful stop) | kept | keep being retried |
| Redis | kept if Redis persists (AOF/RDB on its volume); the api answers `503` meanwhile and workers reconnect | as before | as before | as before |
| Redis without persistence | lost; a client that checks its open jobs gets `404` and submits again | lost | lost | lost |

## Scaling

Add worker containers (or replicas): they all listen on the same Redis queues, so the capacity is
`MAX_CONCURRENT_CONVERSIONS x number of worker containers`. Add api replicas behind a load balancer as needed; they
share no state. Size the worker container for `MAX_CONCURRENT_CONVERSIONS` of your largest files at the same time:
the memory cap is per work-horse (85 % of the container each, since it caps address space, not real use); if the
container is too small the kernel kills one horse (`out_of_memory`) and the others continue. Disk, worst case:
`MAX_CONCURRENT_CONVERSIONS x MAX_FILE_SIZE_BYTES` in the worker's temp dir (plus the previews being sent). Redis holds
a job's markdown for `RESULT_TTL_SECONDS`: up to `MAX_MARKDOWN_CHARS` per job.

## Isolation

The C++ parsers read untrusted files. Every job runs in its own work-horse process (its own process group) with a hard
timeout and an address-space limit; a crash or runaway file ends the job (`unreadable`, `timeout`, `out_of_memory`) and
never affects the worker, the api or other jobs. Downloads are streamed to a temp directory that is removed afterwards.
The file name of a job is never used as a path, and URLs (which carry signatures) are never logged, only their host.

See also [SECURITY.md](../SECURITY.md) for deployment advice.
