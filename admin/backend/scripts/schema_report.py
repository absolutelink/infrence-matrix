"""Report-only schema check (Phase 12 prestart pass).

Run after ``alembic upgrade head``: for every ProviderType with a
committed schema, validate every existing ProviderDefinition's
``backend_config`` of that type and log a WARNING per failing row.
Never mutates, never raises — the output feeds the operator's startup
banner (a schema committed by consensus can retroactively disagree with
configs written before it; the admin API only validates on write).

Skips silently on a fresh install (no ProviderType rows yet).
"""

import logging
import os
import sys

logger = logging.getLogger("admin.schema_report")


def _resolve_engine():
    """M2: honor SQLALCHEMY_URL (what prestart.sh exports) so the report
    runs against the same database the migrations just upgraded; fall back
    to the app engine (POSTGRES_* env) otherwise."""
    from sqlmodel import create_engine

    url = os.environ.get("SQLALCHEMY_URL")
    if url:
        return create_engine(url)
    from app.core.db import engine

    return engine


def main() -> int:
    try:
        import jsonschema
        from sqlmodel import Session, select

        from app.models import ProviderDefinition, ProviderType

        with Session(_resolve_engine()) as session:
            types = session.exec(select(ProviderType).order_by(ProviderType.name)).all()
            if not types:
                logger.info("schema report: no provider types registered — skipping")
                return 0
            failures = 0
            for ptype in types:
                validator = jsonschema.Draft202012Validator(ptype.schema or {})
                definitions = session.exec(
                    select(ProviderDefinition)
                    .where(ProviderDefinition.provider_type == ptype.name)
                    .order_by(ProviderDefinition.alias)
                ).all()
                for definition in definitions:
                    errors = list(
                        validator.iter_errors(definition.backend_config or {})
                    )
                    if errors:
                        failures += 1
                        detail = "; ".join(
                            f"{'$' if not e.absolute_path else '.'.join(str(p) for p in e.absolute_path)}: {e.message}"
                            for e in errors[:10]
                        )
                        more = (
                            f" (+{len(errors) - 10} more)" if len(errors) > 10 else ""
                        )
                        logger.warning(
                            "definition '%s' (type '%s') backend_config does not "
                            "validate against the committed schema %s: %s%s",
                            definition.alias,
                            ptype.name,
                            ptype.schema_fingerprint[:8],
                            detail,
                            more,
                        )
            if failures:
                logger.warning(
                    "schema report: %d definition(s) fail committed-schema "
                    "validation (report-only; see warnings above)",
                    failures,
                )
            else:
                logger.info("schema report: all definitions validate OK")
        return 0
    except Exception as exc:  # never block startup on a bad schema/DB state
        logger.warning("schema report skipped due to error: %s", exc)
        return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
    )
    sys.exit(main())
