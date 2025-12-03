
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def run_boltz_prediction(
    yaml_content: str,
    models_dir: Path,
    output_dir: Path,
    use_msa_server: bool = True,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    clear_previous: bool = True
) -> Path:
    """Run Boltz-2 structure prediction with proper output handling"""
    if clear_previous and output_dir.exists():
        import shutil
        logger.info(f"🗑️  Clearing previous results from {output_dir}")
        shutil.rmtree(output_dir)
        logger.info("   Previous results deleted")

    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = output_dir / "input.yaml"
    input_path.write_text(yaml_content)

    cmd = [
        "boltz", "predict",
        str(input_path),
        "--cache", str(models_dir),
        "--out_dir", str(output_dir),
        "--sampling_steps", str(sampling_steps),
        "--diffusion_samples", str(diffusion_samples),
        "--recycling_steps", str(recycling_steps),
    ]

    if use_msa_server:
        cmd.append("--use_msa_server")

    logger.debug(f"Running Boltz-2 prediction...")
    logger.debug(f"Command: {' '.join(cmd)}")
    logger.debug(f"Output directory: {output_dir}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False
    )

    if result.stdout:
        logger.debug("STDOUT:")
        logger.debug(result.stdout)

    if result.stderr:
        logger.debug("\nSTDERR:")
        logger.debug(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Boltz prediction failed with exit code {result.returncode}. Check error messages above.")

    logger.info("\n✅ Prediction complete")
    logger.debug(f"\nListing all files in output directory:")
    for item in output_dir.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(output_dir)
            size = item.stat().st_size
            logger.debug(f"  {rel_path} ({size:,} bytes)")

    return output_dir