import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

import numpy as np
from sklearn.cluster import *

diss = np.load('diss_xu.npy')

df = pd.read_csv('data/Xu_et_al_2016_dataset.xlsx')
lats=list(df.iloc[0].values) # latitudues
longs=list(df.iloc[1].values) # latitudues
sample_names=list(df)

app = dash.Dash()

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        id='year-slider',
        min=1,
        max=10,
        value=1,
        step=None,
        marks={str(year): str(year) for year in range(1,11)}
    )

])

    # dcc.Graph(
    #     id='graph',
    #     figure={
    #         'data' : [{
    #             'lat': lats, 'lon': longs, 'text': sample_names, 'type': 'scattermapbox',
    #             'mode':'markers', 'marker':dict(size=20, color=[2, 0, 0, 0, 0, 0, 0, 4, 5, 5, 4, 2, 2, 3, 3, 1, 1, 1, 1])
    #         }],
    #         'layout': {
    #                 'mapbox': {
    #                         'accesstoken': (
    #                         'pk.eyJ1IjoiY2hyaWRkeXAiLCJhIjoiY2ozcGI1MTZ3M' +
    #                         'DBpcTJ3cXR4b3owdDQwaCJ9.8jpMunbKjdq1anXwU5gxIw'
    #                     )
    #                 },
    #                 'margin': {
    #                     'l': 0, 'r': 0, 'b': 0, 't': 0
    #                 },
    #             }
    #         }
    # ),


@app.callback(
    dash.dependencies.Output('graph-with-slider', 'figure'),
    [dash.dependencies.Input('year-slider', 'value')])
def update_figure(selected_year):

    no_clusters = selected_year

    model = AgglomerativeClustering(linkage='complete',
                               affinity='precomputed',
                               compute_full_tree=False,
                               n_clusters=no_clusters)

    m = model.fit(diss)

    return {
        'data': [{
            'lat': lats, 'lon': longs, 'text': sample_names, 'type': 'scattermapbox',
            'mode':'markers', 'marker':dict(size=20, color=m.labels_)
        }],
        'layout': go.Layout(
            mapbox={'accesstoken': (
                    'pk.eyJ1IjoiY2hyaWRkeXAiLCJhIjoiY2ozcGI1MTZ3M' +
                    'DBpcTJ3cXR4b3owdDQwaCJ9.8jpMunbKjdq1anXwU5gxIw')},
            xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            margin={'l': 100, 'b': 0, 't': 0, 'r': 0},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }

#{'lat': -92, 'lon': 31}
            # pitch={0},
            # zoom={5},


if __name__ == '__main__':
    app.run_server()
