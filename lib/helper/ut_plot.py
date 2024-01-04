from typing import List, Tuple
import plotly.graph_objects as go

def plot_curve(x_values: List[float], 
                           y_values: List[float]) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = x_values, y = y_values, mode = 'lines'
    ))

    fig.update_layout(
        title='Simple Line Chart',
        xaxis=dict(title='X-axis', showspikes=True, 
                   spikecolor="grey", spikesnap="cursor", 
                   spikemode="across", spikedash="dash"),
        yaxis=dict(title='Y-axis'),
        autosize=True
    )
    fig.update_layout(width=800, height=600)

    return fig