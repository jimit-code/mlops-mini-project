# src/models/model_promote_pointer.py

import json
import logging
from pathlib import Path

STAGING_INFO = Path("reports/staging_model.json")
PRODUCTION_INFO = Path("reports/production_model.json")


def get_logger(name: str = "model_promote_pointer") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        fh = logging.FileHandler("model_promotion.log")
        fh.setLevel(logging.DEBUG)

        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(fmt)
        fh.setFormatter(fmt)

        logger.addHandler(ch)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


logger = get_logger()


def promote_staging_to_production(
    staging_path: Path = STAGING_INFO,
    production_path: Path = PRODUCTION_INFO,
) -> None:
    if not staging_path.exists():
        raise FileNotFoundError(
            f"Staging model info not found. {staging_path.resolve()}"
        )

    with staging_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    production_path.parent.mkdir(parents=True, exist_ok=True)
    with production_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    logger.info(
        "Promoted model to production: run_id=%s, path=%s -> %s",
        info.get("run_id"),
        info.get("model_path"),
        production_path.resolve(),
    )


def main():
    try:
        promote_staging_to_production()
    except Exception as e:
        logger.error("Failed to promote model to production. %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()