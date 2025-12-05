import json
from .prediction import run_boltz_prediction
from .viz import plot_confidence_scores
from ..molstar import preload_molstar, show_molstar
from .results import extract_results
import logging


logger = logging.getLogger(__name__)

preload_molstar()  # Needed to preload Molstar in the notebook environment

def run_prediction_and_viz(b, MODELS_DIR, OUTPUT_DIR, ui_components):
    # Clear previous output before starting new prediction

    custom_yaml_input = ui_components["custom_yaml_input"]
    clear_previous_checkbox = ui_components["clear_previous_checkbox"]
    prediction_output = ui_components["prediction_output"]
    

    prediction_output.clear_output(wait=True)
    
    with prediction_output:
        yaml_content = custom_yaml_input.value.strip()
        clear_previous = clear_previous_checkbox.value  
        
        if not yaml_content:
            print("Error: YAML input is empty")
            return

        print("Starting Boltz-2 prediction...")

        RUN_NAME = "user_prediction"

        result_dir = run_boltz_prediction(
            yaml_content=yaml_content,
            models_dir=MODELS_DIR,
            output_dir=OUTPUT_DIR / RUN_NAME,
            use_msa_server=True,
            sampling_steps=200,
            recycling_steps=1,
            diffusion_samples=1,
            clear_previous=clear_previous,
        )

        results = extract_results(result_dir)

        cif_file = None
        if results['structures']:
            cif_file = results['structures'][0]

        confidence_file = None
        if results['confidence']:
            confidence_file = results['confidence'][0]

        show_structure_and_confidence(cif_file, confidence_file)
        

def show_structure_and_confidence(cif_file: str | None, confidence_file: str | None):
    if cif_file:
        with open(cif_file, 'r') as f:
            cif_content = f.read()

        show_molstar(cif_content, height=600, width=1000, title=f"Structure Visualization: {cif_file}")
        print("\nControls: Click + drag to rotate, scroll to zoom, right-click for options\n")
    else:
        logger.info("\nNo structure file provided for visualization")

    if confidence_file:
        with open(confidence_file, 'r') as f:
            confidence_data = json.load(f)

        plot_confidence_scores(confidence_data)
    else:
        logger.info("\nNo confidence file provided for visualization")