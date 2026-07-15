#!/usr/bin/env bash
# =============================================================================
# AUTO-UPDATE SCRIPT - Runs on Raspberry Pi via cron or systemd timer
# Pulls latest code from GitHub and restarts services if changed
# =============================================================================
set -euo pipefail

REPO_DIR="/opt/cam-pi"
BRANCH="main"
LOG_FILE="/data/cam-pi/logs/auto_update.log"
COMPOSE_FILE="${REPO_DIR}/docker-compose.yml"
LOCK_FILE="/tmp/cam-pi-update.lock"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Prevent concurrent runs
if [ -f "$LOCK_FILE" ]; then
    pid=$(cat "$LOCK_FILE")
    if kill -0 "$pid" 2>/dev/null; then
        log "Update already running (PID $pid), exiting"
        exit 0
    fi
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# Ensure repo exists
if [ ! -d "$REPO_DIR/.git" ]; then
    log "Cloning repository..."
    git clone --depth 1 --branch "$BRANCH" \
        "https://github.com/${GITHUB_REPO:-lsduarte16/centinelacam}.git" "$REPO_DIR"
    cd "$REPO_DIR"
    docker compose -f "$COMPOSE_FILE" up -d --build
    log "Initial deployment complete"
    exit 0
fi

cd "$REPO_DIR"

# Fetch latest
git fetch origin "$BRANCH" --depth 1

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH")

if [ "$LOCAL" = "$REMOTE" ]; then
    log "Already up to date ($LOCAL)"
    exit 0
fi

log "Update available: $LOCAL -> $REMOTE"

# Pull changes
git reset --hard "origin/$BRANCH"

# Check if Docker files changed (full rebuild needed)
CHANGED_FILES=$(git diff --name-only "$LOCAL" "$REMOTE" 2>/dev/null || echo "")

if echo "$CHANGED_FILES" | grep -qE "(Dockerfile|docker-compose|requirements|pyproject)"; then
    log "Infrastructure changed - full rebuild"
    docker compose -f "$COMPOSE_FILE" down
    docker compose -f "$COMPOSE_FILE" up -d --build --remove-orphans
    docker image prune -f
else
    log "Code-only change - restarting services"
    docker compose -f "$COMPOSE_FILE" restart cam-pi
fi

# Verify services are healthy
sleep 10
if docker compose -f "$COMPOSE_FILE" ps | grep -q "unhealthy\|Exit"; then
    log "ERROR: Services unhealthy after update, rolling back"
    git reset --hard "$LOCAL"
    docker compose -f "$COMPOSE_FILE" up -d --build
    exit 1
fi

log "Update successful: now at $(git rev-parse --short HEAD)"
