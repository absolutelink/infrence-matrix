# FastAPI Project - Docker Compose Deployment

> **This guide is inherited from the upstream `fastapi/full-stack-fastapi-template`
> and does not match this repository.** Inference Matrix has **no users, no API
> keys, no `SECRET_KEY`, no `FIRST_SUPERUSER`, and no SMTP/email** configuration
> (the `users`/`api_keys` tables were dropped in migration `b7f2d3e9c1a4` and the
> corresponding settings are absent from `backend/app/core/config.py`). The
> Compose service names in `compose.deploy.yml` (`db`, `backend`, `adminer`) are
> also not defined in this repo's `compose.yml` (`postgres`, `frontend`, `agent`).
> Treat every `SECRET_KEY` / `FIRST_SUPERUSER` / `SMTP_*` instruction below as
> not-applicable, and read the deploy caveat in the Deploy section before running
> any command. Use AGENTS.md "Deployment Debugging" for the actual production
> topology (`matrix-app` container on `core@10.100.2.100`, agents on
> `core@10.100.2.111`).

You can deploy the project to your own remote server with Docker Compose. The deployment configuration includes Traefik to handle HTTPS and route incoming traffic to the application.

## Preparation

* Have a remote server ready and available.
* Configure DNS records pointing to the server for the application domain and any supporting service subdomains you want to expose, such as `fastapi-project.example.com` and `adminer.fastapi-project.example.com`.
* Install and configure [Docker](https://docs.docker.com/engine/install/) on the remote server (Docker Engine, not Docker Desktop).

## Copy the Code

```bash
rsync -av --exclude=".git/" --filter=":- .gitignore" ./ root@your-server.example.com:/root/code/app/
```

The `--filter=":- .gitignore"` option tells `rsync` to use the same ignore rules as Git, excluding files such as the Python virtual environment.

## Configure the Application

### Environment Variables

Set the application domain, project name, and first superuser email:

```bash
export DOMAIN=fastapi-project.example.com
export PROJECT_NAME="Full Stack FastAPI Project"
export FIRST_SUPERUSER=admin@example.com
```

You can also configure these environment variables as needed:

* `SMTP_HOST`: The SMTP server host from your email provider.
* `SMTP_USER`: The SMTP server user.
* `EMAILS_FROM_EMAIL`: The email account used to send emails.
* `SENTRY_DSN`: The DSN for Sentry.

### Secrets

Generate and set secure values for the database password, token signing key, and first superuser password:

```bash
export POSTGRES_PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export FIRST_SUPERUSER_PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

To use an authenticated email provider, also set `SMTP_PASSWORD`.

## Deploy

> **Caveat: the commands below do not currently work in this repo.** The
> `compose.deploy.yml` overrides `db`, `adminer`, and `backend`, none of which
> are defined in `compose.yml` (which uses `postgres`/`frontend`/`agent`), and
> `backend` points at a non-existent `backend/Dockerfile`. This file is inherited
> from the upstream full-stack template and has not been reconciled with the
> renamed services. See the Compose caveat in AGENTS.md.

```bash
cd /root/code/app/
docker compose -f compose.yml -f compose.deploy.yml build
docker compose -f compose.yml -f compose.deploy.yml run --rm backend bash scripts/prestart.sh
docker compose -f compose.yml -f compose.deploy.yml up -d
```

The `compose.deploy.yml` file adds HTTPS and automatic certificate handling to the shared `compose.yml` configuration. Explicitly listing both files excludes the local settings from `compose.override.yml`.

The backend Docker image builds the frontend, so the server does not need Bun or prebuilt frontend files.

## Deploy with GitHub Actions

The included `.github/workflows/deploy-docker-compose.yml` workflow runs the deployment commands on the server when manually triggered from GitHub Actions.

Use a self-hosted runner only for a repository whose contributors and workflow code you trust. GitHub recommends using self-hosted runners with private repositories because workflows execute directly on the runner machine.

### Configure Repository Variables and Secrets

In the repository, go to **Settings** > **Secrets and variables** > **Actions** and add these repository variables:

* `DOMAIN`
* `PROJECT_NAME`
* `FIRST_SUPERUSER`

To enable emails, add these optional repository variables:

* `SMTP_HOST`
* `SMTP_USER`
* `EMAILS_FROM_EMAIL`

To enable Sentry, add the optional `SENTRY_DSN` repository variable.

Add these repository secrets:

* `POSTGRES_PASSWORD`
* `SECRET_KEY`
* `FIRST_SUPERUSER_PASSWORD`

To use an authenticated email provider, add the optional `SMTP_PASSWORD` repository secret.

### Install a Self-Hosted Runner

On the server, create a dedicated user and grant it access to Docker:

```bash
sudo adduser github
sudo usermod -aG docker github
sudo su - github
```

In the GitHub repository, go to **Settings** > **Actions** > **Runners**, select **New self-hosted runner**, choose Linux, and follow the commands GitHub provides to download, configure, and register the runner. Install it in `/home/github/actions-runner`.

After registering the runner, exit the `github` user session and install the runner as a system service:

```bash
exit
cd /home/github/actions-runner
sudo ./svc.sh install github
sudo ./svc.sh start
sudo ./svc.sh status
```

See GitHub's guides for [adding a self-hosted runner](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners) and [configuring the runner as a service](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/configure-the-application?platform=linux).

### Run the Deployment

When the runner is online, open the repository's **Actions** tab, select **Deploy with Docker Compose**, and select **Run workflow**.

## URLs

Replace `fastapi-project.example.com` with your domain.

Application (frontend and API): `https://fastapi-project.example.com`

Interactive API docs: `https://fastapi-project.example.com/docs`

Adminer: `https://adminer.fastapi-project.example.com`
