import json
import logging
from pathlib import Path

CANDIDATE_INFO = Path("reports/model_info.json")        # written by model_evaluation
STAGING_INFO = Path("reports/staging_model.json")       # our logical "Staging" pointer


def get_logger(name: str = "model_staging_pointer") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        fh = logging.FileHandler("model_staging_pointer.log")
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


def promote_to_staging(
    candidate_path: Path = CANDIDATE_INFO,
    staging_path: Path = STAGING_INFO,
) -> None:
    """Copy the latest evaluated model info to a 'Staging' pointer file."""
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate model info not found. {candidate_path.resolve()}"
        )

    with candidate_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    staging_path.parent.mkdir(parents=True, exist_ok=True)
    with staging_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    logger.info(
        "Updated staging model pointer to run_id=%s, path=%s at %s",
        info.get("run_id"),
        info.get("model_path"),
        staging_path.resolve(),
    )


def main():
    try:
        promote_to_staging()
    except Exception as e:
        logger.error("Failed to update staging model pointer. %s", e)
        print(f"Error. {e}")


if __name__ == "__main__":
    main()