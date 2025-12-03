from ipywidgets import widgets
from IPython.display import HTML, display

from .input_sequences import INSULIN_YAML, ANTIBODY_ANTIGEN_YAML


def create_input_ui() -> dict[str, widgets.Widget]:
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


def display_ui(name: str, widgets_dict: dict[str, widgets.Widget]) -> None:
    display(HTML(f"<h3>{name} Structure Prediction Interface</h3>"))
    display(widgets_dict["example_buttons"])
    display(widgets_dict["custom_yaml_input"])
    display(widgets_dict["clear_previous_checkbox"])
    display(widgets_dict["run_button"])
    display(widgets_dict["prediction_output"])