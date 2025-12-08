from IPython.display import HTML, display
from ipywidgets import widgets
import json

from .input_sequences import INSULIN_YAML, ANTIBODY_ANTIGEN_YAML
from .molstar import show_molstar

def create_input_ui():
    
    custom_yaml_input = widgets.Textarea(
        value=INSULIN_YAML,
        description='',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='700px', height='200px')
    )

    clear_previous_checkbox = widgets.Checkbox(
        value=True,
        description='Run new prediction (clear previous results)',
        indent=False,
    )

    run_button = widgets.Button(
        description='Run Prediction',
        button_style='success',
    )

    example_buttons = widgets.HBox([
        widgets.Button(description='Load Insulin Example', button_style='info'),
        widgets.Button(description='Load Antibody-Antigen Example', button_style='info')
    ])

    # Output widget to capture and clear prediction results
    prediction_output = widgets.Output()

    def load_insulin(b):
        custom_yaml_input.value = INSULIN_YAML

    def load_antibody(b):
        custom_yaml_input.value = ANTIBODY_ANTIGEN_YAML

    example_buttons.children[0].on_click(load_insulin)
    example_buttons.children[1].on_click(load_antibody)

    return {
        "custom_yaml_input": custom_yaml_input,
        "clear_previous_checkbox": clear_previous_checkbox,
        "run_button": run_button,
        "example_buttons": example_buttons,
        "prediction_output": prediction_output
    }


def display_ui(name: str, widgets_dict, demo:bool = False) -> None:
    if not demo:
        display(HTML(f"<h3>{name} Structure Prediction Interface</h3>"))
        display(widgets_dict["example_buttons"])
        display(widgets_dict["custom_yaml_input"])
        display(widgets_dict["clear_previous_checkbox"])
        display(widgets_dict["run_button"])
        display(widgets_dict["prediction_output"])
    else:
        display_ui_demo(name)



def display_ui_demo(name: str) -> None:
    display(HTML(f"<h3>{name} Structure Prediction Interface</h3>"))

    # Safely embed YAML inside JS using JSON, to avoid quote/backtick issues
    insulin_yaml_js = json.dumps(INSULIN_YAML)
    antibody_yaml_js = json.dumps(ANTIBODY_ANTIGEN_YAML)

    # Buttons + textarea + controls: all pure HTML+JS
    display(HTML(f"""
        <button onclick="document.getElementById('yaml_input').value = {insulin_yaml_js};">
            Load Insulin Example
        </button>
        <button onclick="document.getElementById('yaml_input').value = {antibody_yaml_js};">
            Load Antibody-Antigen Example
        </button>

        <br><br>

        <textarea id="yaml_input"
                  style="width:700px; height:200px;">{ANTIBODY_ANTIGEN_YAML}</textarea>

        <br>
        <input type="checkbox" id="clear_previous" checked>
        <label for="clear_previous">Run new prediction (clear previous results)</label>
        <br><br>

        <button onclick="alert('This is a static demo export. To run predictions, open the notebook in Jupyter with a live kernel.')">
            Run Prediction
        </button>

        <br><br>
        <div id="prediction_output"></div>
    """))

    # Pre-render Mol* with the antigen example.
    # This runs ONCE at notebook execution time.
    cif_file = "./examples/antigen.cif"
    with open(cif_file, "r") as f:
        cif_content = f.read()

    # Assumes show_molstar takes CIF content as a string and renders into the cell output.
    # That output will be embedded into the nbconvert HTML.
    show_molstar(cif_content)
