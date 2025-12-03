import logging
import shutil
import tempfile
from pathlib import Path
import yaml
import numpy as np

from chai_lab.chai1 import run_inference

from ..molstar import show_molstar

logger = logging.getLogger(__name__)


def yaml_to_fasta(yaml_data: dict) -> str:
    """
    Convert YAML sequence format to FASTA format.
    
    Args:
        yaml_data: Dictionary with 'sequences' key containing list of sequence dicts
        
    Returns:
        FASTA formatted string
    """
    fasta_lines = []
    sequences = yaml_data.get("sequences", [])
    
    for seq_dict in sequences:
        if "protein" in seq_dict:
            protein = seq_dict["protein"]
            seq_id = protein.get("id", ["unknown"])[0]
            sequence = protein.get("sequence", "")
            fasta_lines.append(f">protein|name={seq_id}")
            fasta_lines.append(sequence)
    
    return "\n".join(fasta_lines)


def load_aggregate_score(score_path: Path) -> float:
    """
    Load aggregate score from NPZ file.
    
    Args:
        score_path: Path to scores NPZ file
        
    Returns:
        Aggregate score or negative infinity if not found
    """
    if not score_path.exists():
        return -float('inf')
    
    try:
        scores = np.load(score_path)
        return float(scores.get('aggregate_score', -float('inf')))
    except Exception as e:
        logger.warning(f"Failed to load score from {score_path}: {e}")
        return -float('inf')


def find_best_prediction(output_dir: Path) -> tuple[Path, float]:
    """
    Find the best scoring prediction from output directory.
    
    Args:
        output_dir: Directory containing prediction outputs
        
    Returns:
        Tuple of (best_cif_path, best_score)
        
    Raises:
        FileNotFoundError: If no prediction files found
    """
    cif_files = sorted(output_dir.glob("pred.model_idx_*.cif"))
    
    if not cif_files:
        raise FileNotFoundError("No prediction files found in output directory")
    
    best_cif_path = cif_files[0]
    best_score = -float('inf')
    
    for idx, cif_path in enumerate(cif_files):
        score_path = output_dir / f"scores.model_idx_{idx}.npz"
        score = load_aggregate_score(score_path)
        
        if score > best_score:
            best_score = score
            best_cif_path = cif_path
    
    return best_cif_path, best_score


def run_chai_inference(fasta_content: str, output_dir: Path) -> None:
    """
    Run Chai-1 structure prediction.
    
    Args:
        fasta_content: FASTA formatted sequence data
        output_dir: Directory for prediction outputs
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(fasta_content)
        fasta_path = Path(f.name)
    
    try:
        run_inference(
            fasta_file=fasta_path,
            output_dir=output_dir,
            num_trunk_recycles=3,
            num_diffn_timesteps=200,
            seed=42,
            device="cuda",
            use_esm_embeddings=True
        )
    finally:
        fasta_path.unlink(missing_ok=True)


def visualize_best_structure(output_dir: Path) -> None:
    """
    Load and visualize the top-scoring predicted structure.
    
    Args:
        output_dir: Directory containing prediction outputs
    """
    best_cif_path, best_score = find_best_prediction(output_dir)
    
    with open(best_cif_path, 'r') as f:
        cif_content = f.read()
    
    if best_score > -float('inf'):
        title = f"Chai-1 Prediction (Top Score: {best_score:.3f})"
    else:
        title = "Chai-1 Prediction (Top Ranked)"
    
    show_molstar(cif_content, height=600, width=1000, title=title)
    logger.info(f"Displayed top scoring structure: {best_cif_path.name}")


def run_prediction_and_viz(button, models_dir: Path, output_dir: Path, ui_components: dict) -> None:
    """
    Execute Chai-1 prediction and visualize results.
    
    Args:
        button: Button widget that triggered this function
        models_dir: Path to downloaded model weights
        output_dir: Directory for prediction outputs
        ui_components: Dictionary containing UI widgets
    """
    prediction_output = ui_components["prediction_output"]
    yaml_input = ui_components["custom_yaml_input"]
    clear_previous = ui_components["clear_previous_checkbox"]
    
    with prediction_output:
        prediction_output.clear_output()
        
        try:
            input_data = yaml.safe_load(yaml_input.value)
            
            if clear_previous.value and output_dir.exists():
                logger.info(f"Clearing previous results from {output_dir}")
                shutil.rmtree(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            fasta_content = yaml_to_fasta(input_data)
            
            logger.info("Starting Chai-1 prediction")
            run_chai_inference(fasta_content, output_dir)
            
            cif_count = len(list(output_dir.glob("pred.model_idx_*.cif")))
            logger.info(f"Prediction complete. Generated {cif_count} structures")
            
            visualize_best_structure(output_dir)
        
        except FileNotFoundError as e:
            logger.error(f"Prediction failed: {e}")
            print(f"Error: {e}")
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            print(f"Error during prediction: {str(e)}")