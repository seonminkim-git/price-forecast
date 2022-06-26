# %% 
# IMPORT LIBRARY
import numpy as np
import pandas as pd

import os
import pickle
import glob
import json
import random
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_daq as daq
from dash import Dash, dash_table

import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from datetime import datetime

# %% 
# IMPORT DATA
current_path = os.getcwd()
os.chdir(current_path)
with open('../01. Data/data-raw,corr.pkl', 'rb') as file:
    y, x, corrcoef = pickle.load(file)

df = x.copy()
compare_model = pd.read_csv("../01. Data/dash_compare_model.csv", encoding='cp949')

top5 = pd.read_csv("../01. Data/dash_top5 models.csv", encoding='cp949')
top5_model1 = pd.read_csv("../01. Data/dash_top5_model1.csv", encoding='cp949')
top5_model2 = pd.read_csv("../01. Data/dash_top5_model2.csv", encoding='cp949')
top5_model3 = pd.read_csv("../01. Data/dash_top5_model3.csv", encoding='cp949')
top5_model4 = pd.read_csv("../01. Data/dash_top5_model4.csv", encoding='cp949')
top5_model5 = pd.read_csv("../01. Data/dash_top5_model5.csv", encoding='cp949')

# 사용할 Color setting
category = np.unique([i.split('-')[0] for i in x.columns])

colors = [
    "#001f3f",
    "#0074d9",
    "#3d9970",
    # "#111111",
    "#01ff70",
    "#ffdc00",
    "#ff851B",
    "#ff4136",
    "#85144b",
    "#f012be",
    "#b10dc9",
    "#AAAAAA",
    "#111111",
]

# %%
app = Dash(__name__)

# TAB1 - TABLE
style_data = [
    {
        "if": {"state": "active"},
        "backgroundColor": "rgba(150, 180, 225, 0.2)",
        "border": "1px solid white",
    },
    {
        "if": {"state": "selected"},
        "backgroundColor": "rgba(0, 116, 217, .03)",
        "border": "1px solid white",
    },
]

top5_table = dash_table.DataTable(
    id='show_table',
    data=top5.iloc[:,:6].to_dict('records'),
    style_cell={
        'textAlign': 'center',
         # ‘minWidth’: 95,
         # ‘maxWidth’: 95,
         'width': 110,
         'height':40,
         'color': 'black',
         'fontWeight':'normal',
         'fontSize':15,
         'font_family': 'dosis' , # 'Nanum Pen Script',
     },
    style_header={
        'backgroundColor': '#CCAF98',  # #ABB4B3
        'fontWeight':'bold',
        'color':'black',
        'fontSize': 19,
        'font_family': 'dosis',
        'textAlign': 'center',
    },
    style_data_conditional = style_data,
    style_table={'height': '250px','width':'90%'},
    # fixed_rows={‘headers’: True},
    sort_action='native',
    sort_mode='multi',
    row_deletable=False,
    # style_as_list_view=True,
    editable=True,
)

# TAB1 - TEXT
# text_content = "철근 가격 예측을 위해 특정 기준(단변량/공급small/공급large/전부)으로 구분하고 학습 기간을 최소 3개월~ 최대 10년으로 진행했습니다. 모델 평가지표로는 MAPE와 RMSE를 활용하였으며, 실제값과 예측값을 비교해보았습니다. 자세한 내용들은 해당 페이지의 Tab을 통해 확인할 수 있습니다."

# TAB2 - INPUT(X)
drop_cat = dcc.Dropdown(
    id = 'drop_cat',
    clearable = False,
    searchable = False,
    options = [{'label': category[i], 'value': i} for i in range(len(category))],
    value = 0,
    style={
        "margin": "4px",
        "box-shadow": "0px 0px #ebb36a",
        "border-color": "#546089",
        'margin-top': '15px'
        }
)

checklist_cat = dcc.Checklist(
    value = [0],
    id = 'checklist_cat',
    inline=True,
    style = {'color':'orange', 'margin-right':'10px'}
)

# TAB3
drop_date = dcc.Dropdown(
    id = 'drop_date',
    clearable = False,
    searchable = False,
    options = [{'label': i, 'value': idx} for idx, i in list(enumerate(['3 months','1 year','3 years','5 years','10 years']))],
    value = 0,
    style={
        "margin": "4px",
        "box-shadow": "0px 0px #ebb36a",
        "border-color": "#546089",
        'margin-top': '15px'
        }
)

drop_perform = dcc.RadioItems(
    id = 'drop_perform',
    # className="radio",
    options=[
        dict(label="MAPE", value=0),
        dict(label="RMSE", value=1),
    ],
    value=0,
    inline=True
)

drop_input = dcc.Dropdown(
    id = 'drop_input',
    clearable = False,
    searchable = False,
    options = [{'label': i, 'value': idx} for idx, i in list(enumerate(['단변량','공급_small','공급_large','전부 다']))],
    value = 0,
    style={
        "margin": "4px",
        "box-shadow": "0px 0px #ebb36a",
        "border-color": "#546089",
        'margin-top': '15px'
        }
)

# %%
# 상단 Title
NAVBAR = dbc.Navbar(
    children=[
            html.Div(
                [
                    html.Div([html.Img(src=app.get_asset_url('ci.png'), 
                                       className='ci_image'),
                             dbc.NavbarBrand("SK에코플랜트 철근 가격 예측", className="header_title", style={'display':'inline-block'})])
                ],
                style = {'margin-bottom':'20px'}
            )
    ],
)

# TAB1
tab1 = html.Div(children = [
    # html.Div([html.H3("SK에코플랜트 철근 가격 예측 Summary", style = {'margin':'30px'}),
    #           html.P(text_content, style = {'margin':'35px'})],
    #             className='overview-contents',
    #             style = {'margin-top':'5px','margin-bottom':'5px'}),
    
    html.Div(children = [
        html.Div(children = [
            html.Div(children = [
                html.H3("Top5 models", className='h3', style = {'textAlign': 'center'}),
                top5_table
                ]
                 ),
            html.Div(children = [
                html.H3("Feature Importance", className='h3', style = {'textAlign': 'center'}),
                html.Div(dcc.Graph(id='graph_feature_imp', style={'display': 'inline-block'}, className='graphs')
                         )
                ]
                     )
            ],
                 style={'display': 'inline-block','width':'50%', 'margin-left':'15px'}),
        html.Div(children = [
            html.H3("Best Model Summary", className='h3', style = {'textAlign': 'center'}),            
            html.Div(dcc.Graph(id = 'best_model_graph', className = 'graphs'))
        ],
                 style={'display': 'inline-block', 'width':'50%'})
        ],
             style={'display': 'flex'})
    ])

# TAB2
target_x = html.Div([
            html.Div([
                html.Div(children=[
                    html.H3("Factors", style={'margin-bottom':'3%'}),
                    html.Div(drop_cat, className = 'select-dropdown')],
                         className = "card", style = {'padding':'22px', 'margin-top':'0px','height':'140px'}),
                html.Div(children = [
                    html.H3("Sub Factors", style={'margin-bottom':'3%'}),
                    html.Div(checklist_cat, style = {'padding':'1%'})], 
                         className = "card", style = {'padding':'22px', 'height':'510px'})
                ],
                     style={'display': 'inline-block'}),
            html.Div([
                html.Div([dcc.Graph(id='graph_xy1', style={'display': 'inline-block'}, className='graphs'),
                          dcc.Graph(id='graph_xy2', style={'display': 'inline-block','padding-left':'10px'}, className='graphs')]),
                html.Div([dcc.Graph(id='graph_xy3', style={'display': 'inline-block'}, className='graphs'),
                          dcc.Graph(id='graph_xy4', style={'display': 'inline-block','padding-left':'10px'}, className='graphs')])
                     ],
                    style={'display': 'inline-block', 'margin':'22px'})
            ],
                    style={"display": "flex"})

tab2 = html.Div(children = [
    html.Div(id='target-contents', children=[target_x])
])
    
# TAB3    
tab3 = html.Div(children = [
    html.Div(children = [
        dcc.Store(id='store'),
        html.Div([html.H3("Learning Period", style={'margin-bottom':'10px'}),
                 html.Div(drop_date, className = "select-dropdown")],
                 style={'display': 'inline-block',
                        'width':'500px'},
                 className = 'card'),
        html.Div([html.H3("Learning Input", style={'margin-bottom':'10px'}),
                 html.Div(drop_input, className = "select-dropdown")],
                 style={'display': 'inline-block',
                        'margin-left': '5px',
                        'width':'500px'},
                 className = 'card'),        
        html.Div([html.H3("Performance", style={'margin-bottom':'10px'}),
                 html.Div(drop_perform, className = "select-dropdown")],
                 style={'display': 'inline-block',
                        'margin-left': '5px',
                        'width':'500px'},
                 className = 'card')
    ],
             style = {'display':'flex'}),
    html.Div(children = [
        dcc.Graph(id='graph_compare1', style = {'display':'inline-block'}, className='graphs'),
        dcc.Graph(id="graph_compare2", style = {'display': "inline-block"}, className='graphs')
    ], 
             style = {'margin-left':'15px'})
])

# App Layout
app.layout = dbc.Container(
    [
        html.Div(NAVBAR),
        dcc.Tabs(
            [
                dcc.Tab(id='tab-1', children = [tab1], label="Overview",selected_className="custom-tab--selected",style = {'background-color':'#f8f5f0','color':'gray'}),
                dcc.Tab(id='tab-2', children = [tab2], label="EDA",selected_className="custom-tab--selected",style = {'background-color':'#f8f5f0','color':'gray'}),
                dcc.Tab(id='tab-3', children = [tab3], label="Compare Models",selected_className="custom-tab--selected",style = {'background-color':'#f8f5f0','color':'gray'})
            ],
            className="custom-tabs",
            id = 'tabs'
        )
    ],
    fluid=True
)

# %%
# Callbacks

# Tab1 - update selected table cell style
@app.callback(
    Output("show_table", "style_data_conditional"),
    [Input("show_table", "active_cell")]
)
def update_selected_row_color(active):
    style = style_data.copy()
    if active:
        style.append(
            {
                "if": {"row_index": active["row"]},
                "backgroundColor": "rgba(150, 180, 225, 0.2)",
                "border": "1px solid white",
            },
        )
    return style

# Tab1 - update feature importance graph & predict graph
@app.callback(
    [Output("graph_feature_imp", "figure"),
     Output("best_model_graph", "figure")],
    [Input("show_table","data"),
     Input("show_table","active_cell")]
)
def update_feat_graph(update, active_cell):
    feat_lst = [top5_model1, top5_model2, top5_model3, top5_model4, top5_model5]
    
    num = 0
    if active_cell:
        num = active_cell['row']
        
    df = feat_lst[num]
    df.sort_values("importance", ascending=True, inplace=True)
    
    # feature graph
    fig_feat = go.Figure()    
    fig_feat.add_trace(go.Bar(x=df['importance'], y=df['feature'], orientation='h',
                         marker = {'color':'#FFAE1A'}))
    fig_feat.update_layout(
        width = 750, height = 350,
        xaxis = dict(
            title="Importance",
            linecolor = "#BCCCDC",
            showgrid = True,
            gridwidth = 1, gridcolor = "#FFF"
        ),
        yaxis = dict(
            # tickangle=-45,
            # showticklabels=False,
            tickfont=dict(size=9),
            # title = "Features",
            linecolor = "#BCCCDC",
            showgrid = True,
            gridwidth = 1, gridcolor = "#FFF"
        ),
        plot_bgcolor="rgba(248, 245, 240, 0.7)"
    )
    
    # predict graph
    fig_best = go.Figure()
    fig_best.add_trace(go.Scatter(
        x = y[-12:].index.to_timestamp(), y = y[-12:].values,
        mode = 'markers+lines',
        line = dict(color = "#FFAE1A", width = 3),
        name = "Y"
    ))
    
    fig_best.add_trace(go.Scatter(
        x = y[-12:].index.to_timestamp(), y = top5.iloc[num, 6:],
        mode = 'markers+lines',
        line = dict(color = "#4B8C8C", width = 3),
        name = "Pred Y"
    ))
    
    fig_best.update_layout(width = 800, height = 650,
                           legend_x = 0.1, legend_y = -0.2,
                           legend = dict(orientation='h'),
                           xaxis = dict(
                               title = 'Date',
                               linecolor = "#BCCCDC",
                               showgrid = True,
                               gridwidth = 1, gridcolor = "#FFF"
                           ),
                           yaxis = dict(
                               title = "Price of Y",
                               linecolor = "#BCCCDC",
                               showgrid = True,
                               gridwidth = 1, gridcolor = "#FFF"
                           ),
                           plot_bgcolor = 'rgba(248, 245, 240, 0.7)')
    
    return fig_feat, fig_best
        

# Tab2 - update subfactor list
@app.callback(
    [Output("checklist_cat",'options'),
     Output("checklist_cat",'value')],
    Input("drop_cat", 'value')
)
def update_checklist(dropvalue):
    lst = corrcoef.sort_values('highest corr', ascending=False, key=abs).index
    
    idx = [category[dropvalue] in i for i in lst ]
    checklist = lst[idx]
    
    checklist = ['-'.join(i.split('-')[1:]) for i in checklist]
    options = [{'label': checklist[i], 'value': i} for i in range(len(checklist))]
    return options, [0]

# Tab2 - Show graph - INPUT(X)
@app.callback([Output("graph_xy1", "figure"),
               Output("graph_xy2", "figure"),
               Output("graph_xy3", "figure"),
               Output("graph_xy4", "figure")],
              [Input("drop_cat","options"),
               Input("drop_cat","value"),
               Input("checklist_cat","options"),
               Input("checklist_cat","value")])
def output(category, select1, sub, select2):
    cols = []
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Scatter(x = x.index.to_timestamp(), y = y.values,
                              name = "철근 가격", mode='lines',
                              text= y.values,
                              hovertemplate='%{text}',
                              line = {'color': '#32699C', 'width': 4, 'dash':'dot'}), 
                   secondary_y=False)
    fig2 = go.Figure()
    fig3 = go.Figure()
    fig4 = go.Figure()
    
    for t in range(len(select2)):
        col1 = category[select1]['label']
        col2 = sub[select2[t]]['label']
        col = '-'.join([col1, col2])
        cols.append(col)
        
    tmp = 0    
    for c in cols:
        minmax = (x[c].values - x[c].values.min()) / (x[c].values.max() - x[c].values.min())
        fig1.add_trace(go.Scatter(x = df.index.to_timestamp(), y = minmax,
                                  name = c.split('-')[1], mode='lines',
                                  marker = {'color':colors[tmp]},
                                  text = x[c].values,
                                  hovertemplate='%{text}'+' (scaled: %{y:.2f})'), 
                       secondary_y=True)
        
        fig2.add_trace(go.Scatter(x=corrcoef.columns[:(-2)], y=corrcoef.loc[c,:].values,
                            mode='markers+lines', name = c.split('-')[1],
                            marker = {'color':colors[tmp]}))
        
        fig2.add_vline(x = str(corrcoef.columns[np.argmax(corrcoef.loc[c, :'lag=-13'])]),
                line_width = 3)
        
        fig3.add_trace(go.Scatter(x = df.loc[:, c].values, y=y.values,
                                  name = c.split('-')[1] + "_Corr: " + str(round(corrcoef.loc[c,"lag=0"], 2)), 
                                  mode='markers',
                                  marker = {'color':colors[tmp]}))
        
        
        
        best_lag = corrcoef.loc[c,"best lag"]
        best_corr = corrcoef.loc[c,"highest corr"]
        
        if best_lag > 0:
            best_lag_ = int(best_lag)
            x_ = x[~y.shift(best_lag_).isna()].loc[:,c].values
            y_ = y.shift(best_lag_).dropna().values
        elif best_lag < 0:
            best_lag_ = int(np.abs(best_lag))
            x_ = x.loc[:,c].shift(best_lag_).dropna().values
            y_ = y[~x.loc[:,c].shift(best_lag_).isna()].values
        else:
            x_ = x.loc[:,c].values.copy()
            y_ = y.values.copy()
            
        fig4.add_trace(go.Scatter(x=x_, y=y_, mode="markers", 
                                  name = c.split('-')[1] + "_best lag: " + str(int(best_lag)) + ", corr: " + str(round(best_corr, 2)),
                                  marker = {'color':colors[tmp]}))
        
        tmp += 1
        
    width_, height_ = 600, 370 
    fig1.update_layout(
        legend_x = 0.0, legend_y = -0.4,
        legend = dict(orientation='h'),
        width = width_, height = height_,
        plot_bgcolor="rgba(248, 245, 240, 0.7)",
        hovermode='x unified',
        xaxis = dict(
            title = "연도",
            linecolor = "#BCCCDC",
            showgrid = True,
            gridwidth=1, gridcolor='#FFF'
        ) 
        )
    fig1.update_yaxes(title_text="<b>철근 가격</b>", secondary_y=False,
                      linecolor = "#BCCCDC", showgrid=True, titlefont=dict(color="#32699C"),
                      tickfont = dict(color="#32699C"), gridwidth=1, gridcolor = '#FFF')
    fig1.update_yaxes(title_text="<b>Factors</b>(MinMax Scaled)", secondary_y=True)
    
    fig2.update_layout(legend_x = 0.0, legend_y = -0.55,
                       legend = dict(orientation='h'),
                       width = width_, height = height_,
                       title = "Time lag에 따른 Correlation",
                       xaxis=dict(
                            title="시차",
                            linecolor="#BCCCDC",  
                            showgrid=True,
                            gridwidth=1, gridcolor='#FFF'   
                        ),
                            yaxis=dict(
                            title="Corr",
                            linecolor="#BCCCDC",  
                            showgrid=True,
                            gridwidth=1, gridcolor='#FFF'  
                        ),
                            plot_bgcolor="rgba(248, 245, 240, 0.7)")
    
    fig3.update_layout(legend_x = 0.0, legend_y = -0.4,
                       title = "lag: 0 일 때의 Correlation",
                       legend = dict(orientation='h'),
                       width = width_, height = height_,
                       xaxis=dict(
                           title = "Factor 가격",
                           linecolor="#BCCCDC",  
                           showgrid=True,
                           gridwidth=1, gridcolor='#FFF'
                        ),
                       yaxis=dict(
                           title="철근 가격",
                           linecolor="#BCCCDC",  
                           showgrid=True,
                           gridwidth=1, gridcolor='#FFF' 
                        ),
                       plot_bgcolor="rgba(248, 245, 240, 0.7)")
    
    fig4.update_layout(legend_x = 0.0, legend_y = -0.4,
                       title = "Best lag & Correlation",
                       legend = dict(orientation='h'),
                       width = width_, height = height_,
                       xaxis=dict(
                           title = "Factor 가격",
                           linecolor="#BCCCDC",
                           showgrid=True,
                           gridwidth=1, gridcolor='#FFF' 
                        ),
                       yaxis=dict(
                           title="철근 가격",
                           linecolor="#BCCCDC",  
                           showgrid=True,
                           gridwidth=1, gridcolor='#FFF'   
                        ),
                            plot_bgcolor="rgba(248, 245, 240, 0.7)")
    
    
    return fig1, fig2, fig3, fig4
    

# TAB3 - COMPARE MODELS
@app.callback([Output("graph_compare1", "figure"),
               Output("graph_compare2", "figure")],
              [Input("drop_date","value"),
               Input("drop_input","value"),
               Input("drop_input","options"),
               Input("drop_perform","value")])
def output(dates, drop_input,drop_input_dict, performs):
    tmp = compare_model.copy()
    
    date_dict = {0: '3m', 1: '1y', 2: '3y', 3: '5y', 4:'10y'}
    date_idx = date_dict[dates] # label값 가져오기(3 month, 1 year...)
    tmp = tmp[tmp['date'] == date_idx]   
   
    if performs == 0:
        del tmp['rmse']
        tmp.rename(columns = {'mape':'perform'}, inplace=True)
    elif performs == 1:
        del tmp['mape']
        tmp.rename(columns = {'rmse':'perform'}, inplace=True)
       
        
    fig1 = go.Figure()
    fig2 = go.Figure()
        
    input_idx = drop_input_dict[drop_input]['label']
    t = tmp[tmp['input'] == input_idx]
    t.reset_index(drop=True, inplace=True)

    fig2.add_trace(go.Scatter(x = y[-12:].index.to_timestamp(), y = y[-12:].values,
                              name='Y value', line=dict(color=colors[0], width=5, dash='dot')))
    for c in range(len(t)):
        fig1.add_trace(go.Bar(x = [t.loc[c,'model']],
                              y = [t.loc[c,'perform']],
                              marker_color = colors[c+1]))
        fig2.add_trace(go.Scatter(x = y[-12:].index.to_timestamp(), y=t.iloc[c,4:].values,
                                  mode='lines', name=t.loc[c,'model'], 
                                  marker = {'color': colors[c+1]}))
       
      
    fig1.update_layout(barmode='group',
                       width = 800,
                       height = 450,
                       plot_bgcolor='rgba(248, 245, 240, 0.7)',
                       showlegend=False,
                        hovermode="closest",
                        xaxis=dict(tickangle=-45, 
                                   title="Models",
                                   linecolor = "#BCCCDC",
                                    showgrid=True,
                                    gridwidth=1, gridcolor = '#FFF',
                                    categoryorder='total ascending'),
                        yaxis=dict(title="성능 지표",
                                   linecolor = "#BCCCDC",
                                    showgrid=True,
                                    gridwidth=1, gridcolor = '#FFF'),
                        clickmode="event+select")
    
    
    fig2.update_layout(width = 800,
                       height = 450,
                       legend_x = 0.1, legend_y = -0.4,
                       legend = dict(orientation='h'),
                       showlegend=True,
                        xaxis=dict(tickangle=-45, 
                                   title="Date",
                                   linecolor = "#BCCCDC",
                                    showgrid=True,
                                    gridwidth=1, gridcolor = '#FFF'),
                        yaxis=dict(title="Price Of Y",
                                   linecolor = "#BCCCDC",
                                    showgrid=True,
                                    gridwidth=1, gridcolor = '#FFF'),
                        plot_bgcolor='rgba(248, 245, 240, 0.7)'
                        )
    
    
    return fig1, fig2

 
if __name__ == '__main__':
    # debug=False 옵션으로 "<>"아이콘 제거 가능, debug 필요시 debug=True 로 설정
    app.run_server(debug=True, use_reloader=False)

# %%
