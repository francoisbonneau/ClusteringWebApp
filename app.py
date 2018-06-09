import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

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
                    'x': [1, 2, 3, 4],
                    'y': [4, 1, 3, 5],
                    'text': ['a', 'b', 'c', 'd'],
                    'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
                    'name': 'Trace 1',
                    'mode': 'markers',
                    'marker': {'size': 12}
                }
            ]
        }
    ),
    
    html.H2(children='''
        QC plots
    '''),
    
])

if __name__ == '__main__':
    app.run_server(debug=True)