import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
# import pandas as pd
import base64

app = dash.Dash(__name__)

# Set the background image
background_image = "path_to_file\\color_mixer_image.jpg"

# Load the background image as a base64-encoded string

IMG_PATH = "path_to_file\\color_mixer_image.jpg"
with open(IMG_PATH, 'rb') as f:
    encoded_image = base64.b64encode(f.read()).decode()
# Define the RGB to XYZ transformation matrix
rgb_to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])

# Define the XYZ to RGB transformation matrix
xyz_to_rgb = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                       [-0.9692660,  1.8760108,  0.0415560],
                       [ 0.0556434, -0.2040259,  1.0572252]])
# Define the color options
color_options = {
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Orange": (255, 165, 0),
    "Indigo": (75, 0, 130),
    "Purple": (128, 0, 128),
    "Black": (0, 0, 0),
    "White": (255, 255, 255)
}

# Define the layout of the app
app.layout = html.Div([
    html.Img(src='data:image/jpeg;base64,{}'.format(encoded_image)),
    # html.Img(src=background_image, style={"width": "100%"}),
    html.Div([
        html.H1(children="Color Mixer"),
        html.Label("Select color 1:"),
        dcc.Dropdown(
            id="color1-dropdown",
            options=[{"label": k, "value": str(v)} for k, v in color_options.items()] + [{"label": "Other", "value": "other"}],
            value=str(color_options["Red"]),
        ),
        html.Br(),
        html.Div(id="color1-output"),
        html.Br(),
        html.Label("Select color 2:"),
        dcc.Dropdown(
            id="color2-dropdown",
            options=[{"label": k, "value": str(v)} for k, v in color_options.items()] + [{"label": "Other", "value": "other"}],
            value=str(color_options["Blue"]),
        ),
        html.Br(),
        html.Div(id="color2-output"),
        html.Br(),
        html.Button("Mix Colors", id="mix-colors-button"),
        html.Div(id="mixed-color-output")
    ], style={"position": "absolute", "top": "50%", "left": "50%", "transform": "translate(-50%, -50%)", "text-align": "center"})
])

# Define the callbacks
@app.callback(
    Output("color1-output", "children"),
    Input("color1-dropdown", "value")
)
def display_color1_output(value):
    if value == "other":
        return dcc.Input(id="color1-input", type="text", placeholder="Enter RGB values (comma-separated)")
    else:
        r, g, b = tuple(map(int, value.strip("()").split(",")))
        return html.Div(style={"background-color": f"rgb({r},{g},{b})", "width": "50px", "height": "50px"})

@app.callback(
    Output("color2-output", "children"),
    Input("color2-dropdown", "value")
)
def display_color2_output(value):
    if value == "other":
        return dcc.Input(id="color2-input", type="text", placeholder="Enter RGB values (comma-separated)")
    else:
        r, g, b = tuple(map(int, value.strip("()").split(",")))
        return html.Div(style={"background-color": f"rgb({r},{g},{b})", "width": "50px", "height": "50px"})

@app.callback(
    Output("mixed-color-output", "children"),
    Input("mix-colors-button", "n_clicks"),
    State("color1-dropdown", "value"),
    State("color2-dropdown", "value"),
    State("color1-input", "value"),
    State("color2-input", "value")
)
def mix_colors(n_clicks, color1, color2,color1_input, color2_input):
    
    print(n_clicks, color1, color2, color1_input, color2_input)
    if n_clicks is not None:
        # Parse the color1 and color2 inputs
        if color1 != "other":
            color1_rgb = tuple(map(int, color_options[color1]))
        else:
            color1_rgb = tuple(map(int, color1_input.split(","))) if color1_input else (0, 0, 0)
        if color2 != "other":
            color2_rgb = tuple(map(int, color_options[color2]))
        else:
            color2_rgb = tuple(map(int, color2_input.split(","))) if color2_input else (0, 0, 0)
        
        # Convert the input colors from RGB to CIE XYZ using matrix multiplication
        color1_xyz = np.dot(rgb_to_xyz, color1_rgb)
        color2_xyz = np.dot(rgb_to_xyz, color2_rgb)
        
        # Mix the chromaticity coordinates of each color by taking their average
        mixed_xyz = (color1_xyz + color2_xyz) / 2
        
        # Calculate the XYZ values of the mixed color
        mixed_rgb = np.dot(xyz_to_rgb, mixed_xyz)
        mixed_rgb = np.clip(mixed_rgb, 0, 255).astype(int)
        mixed_hex = "#{:02x}{:02x}{:02x}".format(*mixed_rgb)
        
        # Convert the input colors to hex strings for display in Dash
        color1_hex = "#{:02x}{:02x}{:02x}".format(*color1_rgb)
        color2_hex = "#{:02x}{:02x}{:02x}".format(*color2_rgb)
        
        # Create a Plotly figure to display the input colors and the mixed color
        fig = go.Figure()
        SIZE = 100
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                                 marker=dict(color=color1_hex, size=SIZE
                                 ), name = "Color 1"))
        fig.add_trace(go.Scatter(x=[1], y=[0], mode="markers",
                                 marker=dict(color=color2_hex, size=SIZE), name = "Color 2"))
        fig.add_trace(go.Scatter(x=[0.5], y=[-0.5], mode="markers",
                                 marker=dict(color=mixed_hex, size=SIZE), name = "Mixed Color"))
        fig.update_xaxes(range=[-0.5, 1.5], showticklabels=False, showgrid=False)
        fig.update_yaxes(range=[-1, 1], showticklabels=False, showgrid=False)
        fig.update_layout(height=400, width = 800, margin=dict(l=0, r=0, t=0, b=0))
        
        # Return the plotly figure as a list
        return [dcc.Graph(figure=fig)]

    # If no button has been clicked yet, return an empty list
    return []



# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
