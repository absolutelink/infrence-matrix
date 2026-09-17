---
name: backup-restore
description: Backup and restore database, configurations, and model metadata
---

# Backup & Restore Skill

Use this skill when creating backups, restoring from backups, or managing backup schedules.

## Commands

### Create Backup

```bash
# Full backup (database + configs + metadata)
uv run python -m app.services.backup create --output /backups/backup-$(date +%Y%m%d).tar.gz

# Database only
uv run python -m app.services.backup create --database-only --output backup-db.tar.gz

# Configs only
uv run python -m app.services.backup create --configs-only --output backup-configs.tar.gz

# With compression
uv run python -m app.services.backup create --compress zstd --output backup.tar.zst
```

**Options:**
- `--database-only`: Backup PostgreSQL only
- `--configs-only`: Backup configuration files only
- `--include-models`: Include model metadata (not GGUF files)
- `--compress`: Compression type (gzip, zstd, none)
- `--output`: Output file path

### Restore Backup

```bash
# Restore full backup
uv run python -m app.services.backup restore /backups/backup-20240101.tar.gz

# Restore database only
uv run python -m app.services.backup restore backup.tar.gz --database-only

# Restore configs only
uv run python -m app.services.backup restore backup.tar.gz --configs-only

# Dry run (show what would be restored)
uv run python -m app.services.backup restore backup.tar.gz --dry-run

# Force restore (overwrite existing)
uv run python -m app.services.backup restore backup.tar.gz --force
```

**Options:**
- `--database-only`: Restore PostgreSQL only
- `--configs-only`: Restore configuration files only
- `--dry-run`: Show what would be restored without making changes
- `--force`: Overwrite existing data without confirmation

### List Backups

```bash
# List available backups
uv run python -m app.services.backup list

# Show backup details
uv run python -m app.services.backup info /backups/backup-20240101.tar.gz

# Verify backup integrity
uv run python -m app.services.backup verify /backups/backup-20240101.tar.gz
```

### Delete Backups

```bash
# Delete specific backup
uv run python -m app.services.backup delete /backups/backup-20240101.tar.gz

# Delete old backups (keep last 7)
uv run python -m app.services.backup cleanup --keep 7

# Delete backups older than 30 days
uv run python -m app.services.backup cleanup --older-than 30d
```

### Schedule Backups

```bash
# Enable automatic backups (daily at 2 AM)
uv run python -m app.services.backup schedule --daily --time 02:00

# Weekly backup (Sunday at 3 AM)
uv run python -m app.services.backup schedule --weekly --day sunday --time 03:00

# Disable automatic backups
uv run python -m app.services.backup schedule --disable

# Show current schedule
uv run python -m app.services.backup schedule --show
```

## Python API

```python
from app.services.backup import BackupManager

backup_manager = BackupManager()

# Create backup
backup_path = await backup_manager.create_backup(
    include_database=True,
    include_configs=True,
    include_model_metadata=True,
    compress="gzip",
    output_dir="/backups"
)

# Restore backup
result = await backup_manager.restore_backup(
    backup_path=backup_path,
    restore_database=True,
    restore_configs=True,
    dry_run=False
)

# List backups
backups = await backup_manager.list_backups()
# Returns: [{path, size, created_at, components, compressed}, ...]

# Verify backup
is_valid = await backup_manager.verify_backup(backup_path)

# Delete backup
await backup_manager.delete_backup(backup_path)

# Cleanup old backups
await backup_manager.cleanup_old_backups(
    keep_count=7,
    keep_days=30
)
```

## Backup Components

### Database Backup

**Includes:**
- All PostgreSQL tables
- Schema definitions
- Indexes
- Sequences

**Format:** SQL dump (pg_dump format)

**Size:** Typically 1-100 MB depending on conversation history

### Configuration Backup

**Includes:**
- `.env` file (sanitized - no secrets)
- `/etc/inference-matrix/*.yaml` configs
- Docker Compose files
- Custom certificates (if any)

**Excludes:**
- API keys and secrets (must be set via environment)
- Database passwords
- External service tokens

**Format:** Tarball of config files

### Model Metadata Backup

**Includes:**
- Model registry (names, paths, sizes)
- Model capabilities and tags
- Download history
- Server instance configurations

**Excludes:**
- GGUF model files (too large)
- Cache files

**Format:** JSON export from database

## Backup Strategies

### Daily Backups

```yaml
schedule:
  daily:
    enabled: true
    time: "02:00"  # 2 AM
    compress: gzip
    keep_days: 7
```

### Weekly Full + Daily Incremental

```yaml
schedule:
  weekly:
    enabled: true
    day: sunday
    time: "03:00"
    full_backup: true
    keep_weeks: 4
  
  daily:
    enabled: true
    time: "02:00"
    incremental: true
    keep_days: 7
```

### Retention Policy

```yaml
retention:
  daily: 7      # Keep 7 daily backups
  weekly: 4     # Keep 4 weekly backups
  monthly: 12   # Keep 12 monthly backups
  min_free_gb: 10  # Minimum free space before cleanup
```

## Restore Scenarios

### Full System Restore

```bash
# 1. Stop services
docker compose down

# 2. Restore from backup
uv run python -m app.services.backup restore /backups/backup-20240101.tar.gz

# 3. Verify models exist
uv run python -m app.services.models list

# 4. Restart services
docker compose up -d

# 5. Check health
curl http://localhost:8000/health
```

### Database-Only Restore

```bash
# Restore database without touching configs
uv run python -m app.services.backup restore backup.tar.gz --database-only

# Services continue running with existing configs
```

### Configuration Restore

```bash
# Restore configs after system rebuild
uv run python -m app.services.backup restore backup.tar.gz --configs-only

# Manually set secrets in .env
# Restart services
docker compose restart
```

## Disaster Recovery

### Complete System Rebuild

1. **Install fresh system:**
```bash
# Fresh install of Inference Matrix
git clone <repo>
cd inference-matrix
```

2. **Restore from backup:**
```bash
# Copy backup from old system
scp user@old-server:/backups/backup-latest.tar.gz .

# Restore everything
uv run python -m app.services.backup restore backup-latest.tar.gz
```

3. **Re-download models:**
```bash
# Get model list
uv run python -m app.services.models list --missing

# Re-download missing models
for model in $(uv run python -m app.services.models list --missing --ids); do
  uv run python -m app.services.models download --model-id $model
done
```

4. **Verify and start:**
```bash
# Verify all components
uv run python -m app.services.backup verify-system

# Start services
docker compose up -d
```

## Backup Verification

### Integrity Check

```python
from app.services.backup import verify_backup_integrity

result = verify_backup_integrity("/backups/backup.tar.gz")
# Checks:
# - File exists and is readable
# - Checksum matches
# - Can extract files
# - Database dump is valid SQL
# - Configs are valid YAML/JSON
```

### Test Restore

```bash
# Restore to temporary location for testing
uv run python -m app.services.backup restore backup.tar.gz \
  --dry-run \
  --target-dir /tmp/test-restore

# Verify files
ls -la /tmp/test-restore/

# Cleanup
rm -rf /tmp/test-restore/
```

## Storage Options

### Local Storage

```yaml
storage:
  type: local
  path: /backups
  max_size_gb: 100
```

### Network Storage (NFS/SMB)

```yaml
storage:
  type: nfs
  path: /mnt/nfs/backups
  mount_options: "rw,sync"
```

### Cloud Storage (S3-compatible)

```yaml
storage:
  type: s3
  bucket: inference-matrix-backups
  region: us-east-1
  prefix: backups/
  
  # Credentials via environment
  # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
```

## Monitoring

### Backup Status

```bash
# Check last backup
uv run python -m app.services.backup status

# Output:
# Last backup: 2024-01-01 02:00:00
# Size: 45.2 MB
# Status: success
# Next scheduled: 2024-01-02 02:00:00
```

### Alerts

```yaml
alerts:
  on_failure: true
  on_missing_backup: true
  on_low_disk_space: true
  min_free_gb: 10
  
  notifications:
    - type: email
      to: admin@example.com
    - type: webhook
      url: https://hooks.slack.com/...
```

## Best Practices

1. **Backup daily**: Automatic scheduled backups
2. **Test restores**: Verify backups can be restored
3. **Offsite storage**: Keep copies in different location
4. **Encrypt sensitive data**: Use encrypted storage for backups
5. **Monitor disk space**: Ensure enough space for backups
6. **Document restore process**: Keep runbook updated
7. **Version backups**: Keep multiple versions (7-30 days)
8. **Exclude large files**: Don't backup GGUF models (re-downloadable)

## Troubleshooting

### Backup Fails

**"Disk full"**: Free up space or use external storage
**"Database locked"**: Stop services before backup
**"Permission denied"**: Run as root or with sudo
**"Invalid backup"**: Verify backup file integrity

### Restore Fails

**"Database exists"**: Use --force or drop database first
**"Config conflict"**: Backup existing configs first
**"Model not found"**: Re-download missing models
**"Version mismatch"**: Use same version as backup

## Configuration

Backup config in `/etc/inference-matrix/backup.yaml`:

```yaml
backup:
  enabled: true
  
  # Schedule
  schedule:
    daily:
      enabled: true
      time: "02:00"
    weekly:
      enabled: false
  
  # Retention
  retention:
    keep_days: 7
    keep_count: 10
    min_free_gb: 10
  
  # Storage
  storage:
    type: local
    path: /backups
    compress: gzip
  
  # Components
  include:
    database: true
    configs: true
    model_metadata: true
    cache: false
  
  # Notifications
  notify:
    on_success: false
    on_failure: true
    channels:
      - email
```
