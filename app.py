# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 02:02:08 2018

@author: Dan
"""
import base64
import datetime
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn.cluster import *
import math
from scipy.stats import norm
import dash_table_experiments as dt
from dash.dependencies import Input, Output

from textwrap import dedent

df = pd.read_csv('data/Xu_et_al_2016_dataset.xlsx')
df = df.rename(columns=lambda x: x.strip())
lats=list(df.iloc[0].values)
longs=list(df.iloc[1].values)
sample_names=list(df)

# ============== back end python ===================
def build_KDE(ages,bw,max_age_roundup):

    ages = np.array(ages[~np.isnan(ages)]) #remove nans and convert to array
    x_range = np.arange(0,max_age_roundup,1) #sampled every 1 spacing
    KDE_bandwidths = [bw for i in ages] # set bandwidth for each age

    sum_pdf = x_range-x_range
    for i in range(len(ages)):
        sum_pdf = sum_pdf + norm.pdf(x=x_range,loc=ages[i],scale=KDE_bandwidths[i])
        norm_sum_pdf = sum_pdf / len(ages)
    return norm_sum_pdf

max_age = max(df[2:].max())
max_age_roundup = int(math.ceil(max_age/500.0))*500 # to nearest 500
bw = 30
KDE_df = pd.DataFrame(data=None, columns=df.columns)

for col in df:
    KDE_series = pd.Series(build_KDE(df[col][2:], bw, max_age_roundup))
    KDE_df[col] = KDE_series

def calc_stat(x,y,stat):
    x = np.array(x)
    y = np.array(y)

    if stat=='likeness':
        M = abs(x-y)
        s = sum(M)/2
        return s

    if stat=='similarity':
        if np.array_equal(x,y):
            s = 0
        else:
            S = np.sqrt(x*y)
            s = 1-sum(S)
        return s

    if stat=='R2':
        xmean = np.mean(x)
        ymean = np.mean(y)
        xcov = np.zeros(len(x))
        ycov = np.zeros(len(y))

        for i in range(len(x)):
            xcov[i] = x[i] - xmean
        for i in range(len(x)):
            ycov[i] = y[i] - ymean
        numerator = sum(xcov*ycov)

        sumxcov2 = sum(xcov*xcov)
        sumycov2 = sum(ycov*ycov)
        mult2 = sumxcov2*sumycov2
        denominator = np.sqrt(mult2)

        r = numerator/denominator
        r2 = r*r
        s = 1-r2
        return s

n = len(KDE_df.columns)
diss = np.zeros((n,n))
stat = 'R2'

for i in range(n):
    for j in range(n):
        diss[i,j] = calc_stat(KDE_df.iloc[:,i], KDE_df.iloc[:,j],stat)

# ============== back end python ===================

scl = [0,"rgb(150,0,90)"],[0.125,"rgb(0, 0, 200)"],[0.25,"rgb(0, 25, 255)"],\
[0.375,"rgb(0, 152, 255)"],[0.5,"rgb(44, 255, 150)"],[0.625,"rgb(151, 255, 0)"],\
[0.75,"rgb(255, 234, 0)"],[0.875,"rgb(255, 111, 0)"],[1,"rgb(255, 0, 0)"]

COLORSCALE = [ [0, "B61F45"], [0.15, "rgb(249,210,41)"], [0.3, "rgb(134,191,118)"],
                [0.45, "rgb(37,180,167)"], [0.6, "rgb(17,123,215)"], [0.85, "716E6B"],[1, "rgb(54,50,153)"] ]
"""
test_df = KDE_df[KDE_df.columns[0]]
traces=[]
traces.append(go.Scatter(
        x=range(len(test_df)),
        y=test_df,
        marker={
            'size': 0,
            'line': {'width': 0.5, 'color': 'black'}
        },
        ))
"""
#=======================

app = dash.Dash()

# Boostrap CSS.
app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})

html_center = 'left'
# html_border = 'solid'
html_border = 'none'

# Main Div
app.layout = html.Div([

    html.Div([
    # Title
    html.Div([
        html.H1(children='Geochrono Cluster Fu...n!', style={'text-align':html_center})
    ], className='ten columns'),

    html.Div([
        html.H3(children='Welcome to an interactive clustering web application for geochronological data. The purpose of this application is to track similaties in rock stories. The provided data should present the distribution of measured ages on rock samples picked up on the field.', style={'text-align':html_center, 'color':'darkblue'})
    ], className='ten columns'),

    html.Div([
        html.H4(children="Step one: Import the data. The CSV file must be formatted as:", style={'text-align':html_center}),
    ], className='ten columns'),

    html.Div([
        dcc.Markdown(dedent('''
            * One column per sample
            * first row: latitude
            * second row: longitudes
            * next rows: ages distribution property
           ''')),
   ], className='ten columns'),

    # Import data
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        )
    ], className='twelve columns'),

    # Data info
    html.Div(id='output-data-upload', className='ten columns'),

    # Table
    html.Div([
        html.H4(dt.DataTable(rows=[{}]), style={'display': 'none'})
    ], className='ten columns'),

    #
    html.Div([
        html.H4("Step two: Detrital Geochronology Clustering", style={'text-align':html_center})
    ], className='ten columns'),

    html.Div([
       html.H5(children="Setup the clustering informations. WIP!", style={'text-align':html_center}),
       dcc.RadioItems(
       options=[
           {'label': 'Agglomerative Clustering', 'value':'SiMap'},
           {'label': 'Principal Component Analysis', 'value': 'PCA'},
           {'label': 'K-means', 'value': 'ML'}
       ],
       value='MTL'
       )
   ], className='six columns'),

    # html.Div([
    #     html.H4("", style={'float':'center'})
    # ], className='ten columns'),

    #
    # html.Div([
    #     html.H5("Clustering Value", style={'text-align':html_center}),
    # ], className='ten columns', style={'transform': 'rotate(90deg)'}),

    # html.Div([
    #     html.H2("", style={'float':'center'})
    # ], className='ten columns'),

    ], className='row', style={'border-style': html_border}),

    # Row with map and slider
    html.Div([
        html.Div([
                dcc.Graph(id='graph-with-slider',
                    figure=go.Figure(
                        data=[go.Scattermapbox(lat=[], lon=[])],
                        layout=go.Layout(
                            height=400,
                            mapbox=dict(
                                accesstoken=('pk.eyJ1IjoiYWxpc2hvYmVpcmkiLCJhIjoiY2ozYnM3YTUxMDAxeDMzcGNjbmZyMmplZiJ9.ZjmQ0C2MNs1AzEBC_Syadg'),
                                # center=dict(
                                #     lat=np.mean(30),
                                #     lon=np.mean(40)
                                # ),
                                zoom=4,
                            ),
                            hovermode='closest',
                            margin=dict(r=10, t=10, l=10, b=10)
                        )
                    )
                )
        ], className='ten columns', style={'border-style': html_border, 'height': '400'}),

        # Slider
        html.Div([
            dcc.Slider(
                id='slider',
                min=2,
                max=8,
                step=None,
                marks={str(year): str(year) for year in range(2,9)},
                vertical=True,
            )
        ], className='two columns', style={'border-style': html_border, 'margin-top': '25', 'margin-bottom': '25', 'height': '300'}),

        #
        html.Div([
            html.P("Clustering Value", style={'text-align':'center'}),
        ], className='two columns', style={'margin-top': '0', 'height': '50', 'border-style': 'none'}),#, 'transform': 'rotate(90deg)'}),

    ], className='row', style={'border-style': html_border, 'border-color': 'red'}),

    html.Div([
        html.H4("Step three: Age Density Distribution", style={'text-align':html_center})
    ], className='ten columns'),

    html.Div([
        dcc.Graph(id='qc_kde_plot',
                  figure=go.Figure(
                      data=[],
                      layout=go.Layout(
                          height=400,
                          margin=dict(r=10, t=10, l=10, b=10),
                          hovermode='closest',
                          xaxis=dict(title='Age'),
                          yaxis=dict(title='Density')),
                      )
                  )
             ], className='ten columns'),

    html.Div([
        html.H4("Step four: Party!", style={'text-align':html_center})
    ], className='ten columns'),

    html.Div([
        html.H4("Making your geoclustering less f...ed", style={'text-align':html_center})
    ], className='ten columns'),

], className='ten columns offset-by-one')

# ---------------
# End of html
# ---------------

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # Use the DataTable prototype component:
        # github.com/plotly/dash-table-experiments
        dt.DataTable(rows=df.to_dict('records')),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

# ---------------------------------
# Callbacks
# ---------------------------------

# Upload data
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

# Map
@app.callback(
    dash.dependencies.Output('graph-with-slider', 'figure'),
    [dash.dependencies.Input('slider', 'value')])
def update_figure(k):
    model = AgglomerativeClustering(linkage='complete',
                               affinity='precomputed',
                               compute_full_tree=False,
                               n_clusters=k)
    m = model.fit(diss)

    c= ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, k)]
    c = [c[i] for i in m.labels_]

    return {
        'data': [{
           'lat': lats,
           'lon': longs,
           'text': sample_names,
           'type': 'scattermapbox',
           'customdata': c,
           'mode':'markers', 'marker':dict(size=15,
                                           line = dict(width=10,color='black'),
                                           color=c,
                                           opacity = 0.9)
                                           }],
        'layout': go.Layout(
           height=400,
           mapbox=dict(
                   accesstoken=('pk.eyJ1IjoiYWxpc2hvYmVpcmkiLCJhIjoiY2ozYnM3YTUxMDAxeDMzcGNjbmZyMmplZiJ9.ZjmQ0C2MNs1AzEBC_Syadg'),
                   center=dict(
                       lat=np.mean(lats),
                       lon=np.mean(longs)),
                   zoom=4,
           ),
           hovermode='closest',
           margin=dict(r=10, t=10, l=10, b=10),
           dragmode='select'
        )
    }

# Histogram?
@app.callback(
    dash.dependencies.Output('qc_kde_plot', 'figure'),
    [dash.dependencies.Input('graph-with-slider', 'selectedData')])
def update_figure(selectedData):

    def color_change(inp):
        c = inp.split(',')
        amd = int(c[1][0])+int(9*np.random.rand(1))
        c[1] = str( str(amd) + c[1][1:] )
        amd2 = int(c[2][0])+int(5*np.random.rand(1))
        c[2] = str( str(amd2) + c[2][1:] )
        return ','.join(c)

    c2 = []
    for colors_out in [selectedData['points'][i]['customdata'] for i in range(len(selectedData['points']))]:
        c2.append(color_change(colors_out))

    traces = []
    numerator = 0
    for samp_name in [selectedData['points'][i]['text'] for i in range(len(selectedData['points']))]:
        ys = list(KDE_df[samp_name])
        xs = list(range(len(KDE_df)))
        traces.append(go.Scatter(
            name=samp_name,
            hovertext=samp_name,
            x=xs,
            y=ys,
            mode='lines',
            line = dict(
                width=2.5,
                color = c2[numerator]
                )
            ))
        numerator = numerator + 1

    return {
            'data':traces,
            'layout': go.Layout(
            margin={'l': 50, 'b': 50, 't': 50, 'r': 50},
            hovermode='closest',
            xaxis=dict(title='Age'),
            yaxis=dict(title='Density')),
    }

#app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
#app.css.append_css({"external_url": "//fonts.googleapis.com/css?family=Dosis:Medium"})
#external_css = ["//fonts.googleapis.com/css?family=Dosis:Medium"]

if __name__ == '__main__':
    app.run_server()

#TODO
    # refind missing numbers on slider
    # deploy as web app
    # polish
    # add axis
