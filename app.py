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
    
    html.H2(children='''
        QC plots
    '''),
    
])

if __name__ == '__main__':
    app.run_server(debug=True)