import json
from textwrap import dedent as d

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

app = dash.Dash()

df = pd.read_csv("data/",delimiter=',')
lats=list(df.iloc[1].values) # latitudues
longs=list(df.iloc[2].values) # latitudues
sample_names=list(df)

app.layout = html.Div(children=[
    html.H1(children='Welcome to Interactive Clustering Application data'),

    html.H2(children='''
        Input data
    '''),
    html.H2(children='''
        Clustering parameters
    '''),

    html.H2(children='''
        Displays Mapss
    '''),
    
    dcc.Graph(
        id='basic-interactions',
        figure={
            'data': [
                {
                    'x': lats,
                    'y': longs,
                    'text': sample_names,
                    'name': 'Cluster 1',
                    'mode': 'markers',
                    'marker': {'size': 12}
                }
            ]
        }
    ),
           
    html.H2(children='''
        QC plots
    '''),
    
    html.H2(children='''
        QC plots
    ''')
    
])


if __name__ == '__main__':
    app.run_server(debug=True)