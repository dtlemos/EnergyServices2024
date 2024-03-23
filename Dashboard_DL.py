import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn import  metrics
import plotly.graph_objects as go
from sklearn.feature_selection import SelectKBest # selection method
from sklearn.feature_selection import mutual_info_regression,f_regression # score metric (f_regression)
from sklearn.ensemble import RandomForestRegressor
from sklearn import  linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

#path = 'C:/Users/leopa/Documents/2º Semestre/SEne/Projecto2/'
path = ''

testDataFile = 'forecast_data.csv'
df_test = pd.read_csv(path + testDataFile)


df_train = pd.read_csv(path +'trainingData17_18.csv')

df_test.set_index('Date', inplace=True)
df_test.index = pd.to_datetime(df_test.index)
Z=df_test.values

Y=Z[:,1]
X=Z[:,[2, 4, 7, 10, 12, 24]]

df_rawVar = df_test.iloc[:,[1, 2 ,3 ,4 ,5 ,6]]
new_column_names = {'Central (kWh)': 'Power (kW)', 'temp_C':'Temperature (ºC)' , 'rain_mm/h' : 'Rain (mm)', 'solarRad_W/m2':  'Solar Irradiance (W/m^2)','pres_mbar': 'Pressure (mbar)' , 'windSpeed_m/s': 'Wind Speed (m/s)'}
df_rawVar = df_rawVar.rename(columns = new_column_names)


new_column_names3 = {'Temperature (C)':'Temperature (ºC)' , 'rain_mm/h' : 'Rain (mm)', 
                     'solarRad_W/m2':  'Solar Irradiance (W/m^2)','pres_mbar': 'Pressure (mbar)' 
                     ,'windSpeed_m/s': 'Wind Speed (m/s)', 'Power-1': 'Power (-1)',
                     'holiday_weekend': 'Holidays/Weekends', 'hour':'Hour','hour_sin':'Hour Sin',
                     'hour_cos': 'Hour Cos','Power (kW)_rolling_mean_2H': 'Power RM-2H',
                     'Power (kW)_rolling_mean_4H': 'Power RM-4H', 'Temperature (C)_rolling_mean_2H':'Temperature RM-2H',
                     'Temperature (C)_rolling_mean_4H':'Temperature RM-4H',
                    'solarRad_W/m2_rolling_mean_2H': 'Solar Irradiance RM-2H',
                     'solarRad_W/m2_rolling_mean_4H': 'Solar Irradiance RM-4H',
                     'Power (kW)_rolling_std_2H':  'Power RStd-2H',
                     'Power (kW)_rolling_std_4H':  'Power RStd-4H',
                     'Temperature (C)_rolling_std_2H': 'Temperature RStd-2H',
                     'Temperature (C)_rolling_std_4H': 'Temperature RStd-4H',
                     'solarRad_W/m2_rolling_std_2H': 'Solar Irradiance RStd-2H',
                     'solarRad_W/m2_rolling_std_4H': 'Solar Irradiance RStd-4H',
                     'Power (kW)_diff': 'Power 1st Derivative','Power (kW)_2nd_diff':'Power 2nd Derivative'}

df_train = df_train.rename(columns = new_column_names3)

f_options = ['Temperature (ºC)' , 'Rain (mm)', 'Solar Irradiance (W/m^2)','Pressure (mbar)',
              'Wind Speed (m/s)','Power (-1)','Holidays/Weekends','Hour','Hour Sin',
              'Hour Cos', 'Power RM-2H','Power RM-4H', 'Temperature RM-2H',
              'Temperature RM-4H','Solar Irradiance RM-2H','Solar Irradiance RM-4H',
              'Power RStd-2H','Power RStd-4H','Temperature RStd-2H','Temperature RStd-4H',
              'Solar Irradiance RStd-2H','Solar Irradiance RStd-4H','Power 1st Derivative', 'Power 2nd Derivative']
m_options = ['Linear Regression', 'Forest Regressor', 'Neural Networks', 'Decision Tree Regressor']


df_features=df_test.copy()
new_column_names2 = {'temp_C':'Temperature (ºC)' , 'rain_mm/h' : 'Rain (mm)', 
                     'solarRad_W/m2':  'Solar Irradiance (W/m^2)','pres_mbar': 'Pressure (mbar)' 
                     ,'windSpeed_m/s': 'Wind Speed (m/s)', 'Power-1': 'Power (-1)',
                     'holiday_weekend': 'Holidays/Weekends', 'hour':'Hour','hour_sin':'Hour Sin',
                     'hour_cos': 'Hour Cos', 'Central (kWh)_rolling_mean_2H': 'Power RM-2H',
                     'Central (kWh)_rolling_mean_4H': 'Power RM-4H', 'temp_C_rolling_mean_2H':'Temperature RM-2H',
                     'temp_C_rolling_mean_4H':'Temperature RM-4H',
                     'solarRad_W/m2_rolling_mean_2H': 'Solar Irradiance RM-2H',
                     'solarRad_W/m2_rolling_mean_4H': 'Solar Irradiance RM-4H',
                     'Central (kWh)_rolling_std_2H':  'Power RStd-2H',
                     'Central (kWh)_rolling_std_4H':  'Power RStd-4H',
                     'temp_C_rolling_std_2H': 'Temperature RStd-2H',
                     'temp_C_rolling_std_4H': 'Temperature RStd-4H',
                     'solarRad_W/m2_rolling_std_2H': 'Solar Irradiance RStd-2H',
                     'solarRad_W/m2_rolling_std_4H': 'Solar Irradiance RStd-4H',
                     'Central (kWh)_diff': 'Power 1st Derivative', 'Central (kWh)_2nd_diff':'Power 2nd Derivative'}


df_features = df_features.rename(columns = new_column_names2)

X2=df_features.values
 
# FIGURE 1

fig1 = px.line(df_rawVar, x=df_rawVar.index, y=df_rawVar.columns[0])# Creates a figure with the raw data
layout = go.Layout(
    title='2019 Power Consumption and Meteorological Dataset',
    yaxis=dict(title=df_rawVar.columns[0], side='left', anchor='x', showgrid=True),
    showlegend=True,
)

fig1.update_layout(layout)

# FIGURE 3

fig3 = px.scatter(df_test, x='temp_C', y='Central (kWh)', title='Scatter Plot', 
                  labels={'temp_C': 'Temperature (C)', 'Central (kWh)': 'Power (kW)'})
fig3.update_layout(xaxis_title='Temperature (C)', yaxis_title='Power (kW)', width = 600,height = 500)

# FIGURE 4

fig4 = go.Figure()
fig4.add_trace(go.Box(y=df_rawVar['Temperature (ºC)'],name = ''))
fig4.update_layout(title='Box Plot', yaxis_title='Temperature (ºC)',width = 600,height = 500)

# FIGURE 5
Y5 = np.array(df_rawVar.iloc[:, 0])
X5 = np.array(df_rawVar.iloc[:, 1:])

features=SelectKBest(k=3,score_func=f_regression)
fit=features.fit(X5,Y5) #calculates the scores using the score_function f_regression of the features

x = [i for i in range(len(fit.scores_))]
x_best = [i for i in fit.get_support(indices=True)]
scores = fit.scores_
columns = np.array(df_rawVar.columns[1:])

fig5 = go.Figure()
fig5.add_trace(go.Bar(x=x, y=scores, name='All Features'))
fig5.add_trace(go.Bar(x=x_best, y=scores[x_best], name='Best Features'))

fig5.update_layout(
    title='With f_regression score',
    xaxis=dict(tickvals=x, ticktext=columns, tickangle=90),
    yaxis=dict(type='log'),
    barmode='overlay',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    width = 600,
    height = 500,
)

# FIGURE 6
Y6 = np.array(df_rawVar.iloc[:, 0])
X6 = np.array(df_features.iloc[:, 1:])

features=SelectKBest(k=5,score_func=f_regression)
fit=features.fit(X6,Y6) #calculates the scores using the score_function f_regression of the features

x = [i for i in range(len(fit.scores_))]
x_best = [i for i in fit.get_support(indices=True)]
scores = fit.scores_
columns = np.array(df_features.columns[1:])

fig6 = go.Figure()
fig6.add_trace(go.Bar(x=x, y=scores, name='All Features'))
fig6.add_trace(go.Bar(x=x_best, y=scores[x_best], name='Best Features'))

fig6.update_layout(
    title='With f_regression score',
    xaxis=dict(tickvals=x, ticktext=columns, tickangle=90),
    yaxis=dict(type='log'),
    barmode='overlay',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    width = 800,
    height = 600,
)

# FIGURE 7

fig7 =  go.Figure()


Y = df_test['Central (kWh)'].values
df_real = df_test['Central (kWh)'] 


#Evaluate errors
def calcMetrics(Y, Y_pred):
    MAE=metrics.mean_absolute_error(Y,Y_pred)
    MBE=np.mean(Y-Y_pred) 
    MSE=metrics.mean_squared_error(Y,Y_pred)  
    RMSE= np.sqrt(metrics.mean_squared_error(Y,Y_pred))
    cvRMSE=RMSE/np.mean(Y)
    NMBE=MBE/np.mean(Y)     
    return MAE,MBE,MSE,RMSE,cvRMSE, NMBE

# Define auxiliary functions

def generate_table(dataframe, max_rows=10, font_size=18):
    dataframe_rounded = dataframe.round(3)
    
    # Find the minimum value of each column (excluding the first column)
    min_values = dataframe_rounded.iloc[:, 1:].min()
    
    # Create the HTML table
    table = html.Table([
        html.Thead(
            html.Tr([html.Th(col, style={'border': '1px solid black', 'padding': '8px', 'font-size': font_size}) for col in dataframe_rounded.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(
                    dataframe_rounded.iloc[i][col],
                    style={
                        'border': '1px solid black',
                        'padding': '8px',
                        'font-size': font_size,
                        'color': 'white' if col != dataframe_rounded.columns[0] and dataframe_rounded.iloc[i][col] == min_values[col] else 'black',
                        'background-color': '#2780E3' if col != dataframe_rounded.columns[0] and dataframe_rounded.iloc[i][col] == min_values[col] else 'white'  # Highlight minimum values
                    }
                ) for col in dataframe_rounded.columns
            ], style={'border': '1px solid black'}) for i in range(min(len(dataframe_rounded), max_rows))
        ])
    ], style={'border-collapse': 'collapse', 'margin': 'auto'})
    
    return table


'''
    PARAGRAPHS
'''
# P1
p1 = '''IST Energy Forecast is a tool developed in the 2023/2024 Energy Services Course, with the objective of predicting 
    the energy consumption of the IST's Central Building for the first months of 2019. To achieve this several models
    are employed, each of which was trained using a combination of meteorological and historical
    consumption time series from 2017 and 2018. This dashboard enables the user to explore the raw data 
    which was used to train the models - first tab 'Raw Data' - and to compare the results of obtained with
    different models - 'Forecast'.
'''

# P2 - IST Raw Data
p2 = ''' Here you can explore the real data-set, that we will try to model. It consists of power consumption data - Power (kW) and 
meteorological data - Temperature (ºC), Rain (mm), Solar Irradiance (W/m^2), Pressure (mbar) and Windspeed (m/s),
from the 1st of January 2019 to the 11th of April 2019. Feel free to modify the temporal range by 
choosing the beginning and ending dates, as well as the features you would want to see, 
using the dropdown menu below. The goal is to determine which characteristics may be more closely associated 
with power usage.
'''

# P3 - Correlation plots
p3 = ''' In this section you are able to visualize the meteorological data against the power consumption records,
in order to identify possible correlations. 
 '''

# P4 - Box PLots
p4 = ''' In order to get an idea of the dispersion of data and identify possible outliers, box plots are useful.
Select different features with the dropdown menu below, in order to visualize the respective box-plot. 
'''

# P5 
p5 = ''' To quantify the predictive importance of each meteoreological feature one can apply different methods
that assign scores to each of them. Such methods include k-Best, which uses as score functions F-value and Mutual Information
and using a Random Forest Regressor. The first is usually faster. In the graph below the different features are ranked. The 
best three features are colored red.
'''

# P6 - Feature Engineering
p6 = ''' One important step in developing models is feature engineering. After identifying the most relevant meteorological
freatures, combinations or transformations applied to these features can be useful for modelling. Several features were engineered. These include:'''

items = [
    "Power (-1) - information of the previous hour power consumption is given to the model;",
    "Holiday/Weekend - its 1 when is holiday or weekend day",
    "Hour Sin/Cos - transorming the hour of the day according to trigonometric function (periodic features can perform better)",
    "[Feature] RM - 2H - corresponds to a rolling mean average of the previous 2 hours",
    "[Feature] RM - 4H - corresponds to a rolling mean average of the previous 4 hours",
    "[Feature] RStd - 2H - corresponds to a rolling standard deviation average of the previous 2 hours",
    "[Feature] RStd - 4H - corresponds to a rolling standard deviation average of the previous 4 hours",
]

p6_cont = '''Select different feature selection methods and see which are the best features suggested by them. Be aware 
that the Forest Regressor can take a while to process (these are many features to consider). The 5-best features are colored
red.       
'''

# P7 - Model
p7 = '''Here you can explore how models created using different features perform. The models are trained with 2017-2019 data
and tested with 2019 data. Use the first dropdown menu to choose the 
model type(s) and the second dropdown menu to select the features use to generate these models. You will see a graph 
representing the real data against the predictions. Below a table is shown which displays different metrics 
which evaluate model performance. 
'''


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Raw Data", id="raw-data-link", href="#", style={"margin": "0 20px"})),
        dbc.NavItem(dbc.NavLink("Forecast", id="forecast-link", href="#", style={"margin": "0 20px"})),
    ],
    brand="IST Energy Forecast Tool",
    brand_href="#",
    color="primary",
    dark=True,
    #className="py-3",  # Increase padding to adjust the height of the navbar
    brand_style={"fontSize": "40px"},  # Adjust the font size of the brand
    className="navbar-fixed"
)


# Center align the nav items
navbar_style = {
    "display": "flex",
    "justify-content": "center",
    "align-items": "center",
    "margin-right": "auto",
}


app.layout = html.Div([
    navbar,
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              [Input('raw-data-link', 'n_clicks'),
               Input('forecast-link', 'n_clicks')])
def render_content(raw_data_clicks, forecast_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div([
            
            html.P(p1, style={"padding-top": "140px"}),
            html.H3('IST Raw Data', style={"padding-top": "10px"}),
            
            html.P(p2,style={"padding-top": "10px"}),
            html.Div(html.Div(style={'padding': '5px'})),
            dcc.Dropdown(['Power (kW)','Temperature (ºC)','Rain (mm)','Solar Irradiance (W/m^2)','Pressure (mbar)','Wind Speed (m/s)',], 'Power (kW)', multi=True
                         ,id = 'Features'),
            html.Div(html.Div(style={'padding': '5px'})),
            dcc.DatePickerRange(
                id='date-picker-range',  # ID for the date picker range component
                start_date=df_rawVar.index.min(),  # Minimum date from the DataFrame
                end_date=df_rawVar.index.max(),  # Maximum date from the DataFrame
                display_format='DD-MM-YYYY'
                ),
            html.Div(html.Div(style={'padding': '5px'})),
            dcc.Graph(
                id='yearly-data',
                figure=fig1,
            ),
            html.H3('Correlation between features and power consumption'),
            html.P(p3,style={"padding-top": "10px"}),
            html.Div(html.Div(style={'padding': '5px'})),
            
            dcc.Dropdown(['Temperature (ºC)','Rain (mm)','Solar Irradiance (W/m^2)','Pressure (mbar)','Wind Speed (m/s)',], 'Temperature (ºC)', multi=False
                         ,id = 'Features2'),
            
            dcc.Graph(
                id='correlation',
                figure=fig3,
            ),
            
            html.H3('Box-plots'),
        
            html.P(p4,style={"padding-top": "10px"}),
            html.Div(html.Div(style={'padding': '5px'})),

            dcc.Dropdown(['Temperature (ºC)','Rain (mm)','Solar Irradiance (W/m^2)','Pressure (mbar)','Wind Speed (m/s)',], 'Temperature (ºC)', multi=False
                         ,id = 'Features3'),
            dcc.Graph(
                id='box-plot',
                figure=fig4,
            ),
            
            html.H3('Feature Selection Methods'),
            html.P(p5,style={"padding-top": "10px"}),
            
            dcc.Dropdown(['kBest-F-Value','kBest-MI','Forest-Regressor',], 'kBest-F-Value', multi=False
                         ,id = 'Feature-Methods'),
            
            html.Div(html.Div(style={'padding': '20px'})),
        
            dbc.Spinner(html.Div(id="loading-output-feat")),
            
            html.Div(html.Div(style={'padding': '10px'})),
            
            dcc.Graph(
                id='feature-selection',
                figure=fig5,
            ),
            html.H3('Feature Engineering'),
            html.P(p6,style={"padding-top": "10px"}),
            
            html.Ul([
                html.Li(items[0]),
                html.Li(items[1]),
                html.Li(items[2]),
                html.Li(items[3]),
                html.Li(items[4]),
                html.Li(items[5]),
                html.Li(items[6])
            ], style={'padding-left': '20px'}),
            
            html.P(p6_cont,style={"padding-top": "10px"}),
            
            dcc.Dropdown(['kBest-F-Value','kBest-MI','Forest-Regressor',], 'kBest-F-Value', multi=False
                         ,id = 'Feature-Methods2'),
            
            html.Div(html.Div(style={'padding': '20px'})),
        
            dbc.Spinner(html.Div(id="loading-output")),
            
            html.Div(html.Div(style={'padding': '10px'})),
            dcc.Graph(
                id='feature-engine',
                figure=fig6,
            ),
        ], style={'width': '80%', 'margin': 'auto'})
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'raw-data-link':
            return html.Div([
                
                html.P(p1, style={"padding-top": "140px"}),
                html.H3('IST Raw Data', style={"padding-top": "10px"}),
                
                html.P(p2,style={"padding-top": "10px"}),
                html.Div(html.Div(style={'padding': '5px'})),
                dcc.Dropdown(['Power (kW)','Temperature (ºC)','Rain (mm)','Solar Irradiance (W/m^2)','Pressure (mbar)','Wind Speed (m/s)',], 'Power (kW)', multi=True
                             ,id = 'Features'),
                html.Div(html.Div(style={'padding': '5px'})),
                dcc.DatePickerRange(
                    id='date-picker-range',  # ID for the date picker range component
                    start_date=df_rawVar.index.min(),  # Minimum date from the DataFrame
                    end_date=df_rawVar.index.max(),  # Maximum date from the DataFrame
                    display_format='DD-MM-YYYY'
                    ),
                html.Div(html.Div(style={'padding': '5px'})),
                dcc.Graph(
                    id='yearly-data',
                    figure=fig1,
                ),
                html.H3('Correlation between features and power consumption'),
                html.P(p3,style={"padding-top": "10px"}),
                html.Div(html.Div(style={'padding': '5px'})),
                
                dcc.Dropdown(['Temperature (ºC)','Rain (mm)','Solar Irradiance (W/m^2)','Pressure (mbar)','Wind Speed (m/s)',], 'Temperature (ºC)', multi=False
                             ,id = 'Features2'),
                
                dcc.Graph(
                    id='correlation',
                    figure=fig3,
                ),
                
                html.H3('Box-plots'),
            
                html.P(p4,style={"padding-top": "10px"}),
                html.Div(html.Div(style={'padding': '5px'})),

                dcc.Dropdown(['Temperature (ºC)','Rain (mm)','Solar Irradiance (W/m^2)','Pressure (mbar)','Wind Speed (m/s)',], 'Temperature (ºC)', multi=False
                             ,id = 'Features3'),
                dcc.Graph(
                    id='box-plot',
                    figure=fig4,
                ),
                
                html.H3('Feature Selection Methods'),
                html.P(p5,style={"padding-top": "10px"}),
                
                dcc.Dropdown(['kBest-F-Value','kBest-MI','Forest-Regressor',], 'kBest-F-Value', multi=False
                             ,id = 'Feature-Methods'),
                
                html.Div(html.Div(style={'padding': '20px'})),
            
                dbc.Spinner(html.Div(id="loading-output-feat")),
                
                html.Div(html.Div(style={'padding': '10px'})),
                
                dcc.Graph(
                    id='feature-selection',
                    figure=fig5,
                ),
                html.H3('Feature Engineering'),
                html.P(p6,style={"padding-top": "10px"}),
                
                html.Ul([
                    html.Li(items[0]),
                    html.Li(items[1]),
                    html.Li(items[2]),
                    html.Li(items[3]),
                    html.Li(items[4]),
                    html.Li(items[5]),
                    html.Li(items[6])
                ], style={'padding-left': '20px'}),
                
                html.P(p6_cont,style={"padding-top": "10px"}),
                
                dcc.Dropdown(['kBest-F-Value','kBest-MI','Forest-Regressor',], 'kBest-F-Value', multi=False
                             ,id = 'Feature-Methods2'),
                
                html.Div(html.Div(style={'padding': '20px'})),
            
                dbc.Spinner(html.Div(id="loading-output")),
                
                html.Div(html.Div(style={'padding': '10px'})),
                dcc.Graph(
                    id='feature-engine',
                    figure=fig6,
                ),
            ], style={'width': '80%', 'margin': 'auto'})

        elif button_id == 'forecast-link':
            return html.Div([
                html.H4('IST Electricity Forecast',style={"padding-top": "150px"}),
                html.P(p7, style = {'padding': '10px'}),
            
                html.Div(html.Div(style={'padding': '5px'})),
               
                dcc.Dropdown(m_options, m_options[:1], multi=True
                             ,id = 'model-opt'),
                
                html.Div(html.Div(style={'padding': '5px'})),
               
                dcc.Dropdown(f_options, f_options[:2], multi=True
                             ,id = 'features-opt'),
                html.Div(html.Div(style={'padding': '5px'})),
                dcc.DatePickerRange(
                    id='date-picker-range2',  # ID for the date picker range component
                    start_date=df_rawVar.index.min(),  # Minimum date from the DataFrame
                    end_date=df_rawVar.index.max(),  # Maximum date from the DataFrame
                    ),
                
                html.Div(html.Div(style={'padding': '15px'})),
            
                dbc.Spinner(html.Div(id="loading-output-models")),
                
                
                html.Div(html.Div(style={'padding': '10px'})),
                
                dcc.Graph(
                    id='models-features-plot',
                    figure=fig7,
                    ),
                
                html.Div(id='table-container'),
                
                html.Div(html.Div(style={'padding': '50px'})),
                # Your forecast components here
            ],style={'width': '80%', 'margin': 'auto', "position": "relative"})
        
        
@app.callback(
    Output('yearly-data', 'figure'),  # Output component: graph
    [Input('date-picker-range', 'start_date'),  # Input component: start date
     Input('date-picker-range', 'end_date'),  # Input component: end date
     Input('Features', 'value')]  # Input component: selected features
)

def update_graph(start_date, end_date, selected_features):
    # Filter the DataFrame based on the selected date range
    filtered_df = df_rawVar[(df_rawVar.index >= start_date) & (df_rawVar.index <= end_date)]
    
    # Create an empty figure
    fig = go.Figure()
    
    if not isinstance(selected_features, list):
        selected_features = [selected_features]
    
    # Add trace for each selected feature with its own y-axis
    for i, feature in enumerate(selected_features):
        fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df[feature],
                                 mode='lines', name=feature,
                                 yaxis=f"y{i+1}"))
    
    # Update layout with y-axes for each feature
    layout = go.Layout(
        title='2019 Power Consumption and Meteorological Dataset',
        yaxis=dict(title='', side='left', anchor='x', showgrid=True, showticklabels = False),
        showlegend=True,

    )
    
    # Add y-axes for additional features
    for i, feature in enumerate(selected_features[1:], start=2):
        layout[f"yaxis{i}"] = dict(title='', side='right', overlaying='y', anchor='x',  showgrid=False, showticklabels=False)
    
    # Update figure layout
    fig.update_layout(layout)
    
    return fig
        
@app.callback(Output('correlation', 'figure'),
              Input('Features2', 'value'))

def update_scatter_plot(selected_feature):
    # Create the scatter plot with the selected feature on the x-axis
    fig3 = px.scatter(df_rawVar, x=selected_feature, y=df_test['Central (kWh)'], title='2019 Data Scatter Plot',
                      labels={selected_feature: f'{selected_feature} (units)', 'Central (kWh)': 'Power (kW)'})
    
    # Update layout
    fig3.update_layout(xaxis_title=f'{selected_feature}', yaxis_title='Power (kW)', width=600, height=500)
    
    return fig3


@app.callback(Output('box-plot', 'figure'),
              Input('Features3', 'value'))

def update_box_plot(selected_feature):
    
    fig4 = go.Figure()
    fig4.add_trace(go.Box(y=df_rawVar[selected_feature],name = ''))
    
    # Update layout
    fig4.update_layout(yaxis_title=f'{selected_feature}', width=600, height=500)
    
    return fig4

@app.callback([Output('feature-selection', 'figure'),Output("loading-output-feat", "children")],
              Input('Feature-Methods', 'value'))

def update_feature_plot(featSelect_method):
    
    Y5 = np.array(df_rawVar.iloc[:, 0])
    X5 = np.array(df_rawVar.iloc[:, 1:])
    
    
    if featSelect_method == 'kBest-F-Value':
        features=SelectKBest(k=3,score_func=f_regression)
        fit=features.fit(X5,Y5) #calculates the scores using the score_function f_regression of the features
        
        x = [i for i in range(len(fit.scores_))]
        x_best = [i for i in fit.get_support(indices=True)]
        scores = fit.scores_
        columns = np.array(df_rawVar.columns[1:])

        
    elif featSelect_method == 'kBest-MI':
        features=SelectKBest(k=3,score_func=mutual_info_regression)
        fit=features.fit(X5,Y5)
        
        x = [i for i in range(len(fit.scores_))]
        x_best = [i for i in fit.get_support(indices=True)]
        scores = fit.scores_
        columns = np.array(df_rawVar.columns[1:])

        
    elif featSelect_method == 'Forest-Regressor':
        model = RandomForestRegressor()
        model.fit(X5, Y5)
        
        
        scores = model.feature_importances_
        x = [i for i in range(len(scores))]
        y_best  = np.sort(scores)[-3:]
        x_best=[]
        for i in range(len(scores)):
            for y in y_best:
                if y==scores[i]:
                    x_best.append(i)
        columns = np.array(df_rawVar.columns[1:])
        
    loading_output2 = None
    
    # Your plotting code here
    fig = None
    
    if featSelect_method:
        # Create a bar plot
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=scores, name='All Features'))
        fig.add_trace(go.Bar(x=x_best, y=scores[x_best], name='Best Features'))
    
        # Update layout
        fig.update_layout(
            title='With ' + featSelect_method  + ' score',
            xaxis=dict(tickvals=x, ticktext=columns, tickangle=90),
            yaxis=dict(type='log'),
            barmode='overlay',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            width = 600,
            height = 500,
        )
        
    else:
        # Display the loading spinner while the graph is being updated
        loading_output2 = dbc.Spinner(
            color="primary",
            children="Loading...",
            style={"position": "absolute", "top": "50%", "left": "50%", "transform": "translate(100%, 100%)"},
            )
        
    return fig, loading_output2

@app.callback([Output('feature-engine', 'figure'),Output("loading-output", "children")],
              Input('Feature-Methods2', 'value'))

def update_feature_plot2(featSelect_method):
    
    Y6 = np.array(df_rawVar.iloc[:, 0])
    X6 = np.array(df_features.iloc[:, 2:])
    print((df_features.iloc[:, 2:].columns))
    print(df_rawVar.columns)
          

    if featSelect_method == 'kBest-F-Value':
        features=SelectKBest(k=5,score_func=f_regression)
        fit=features.fit(X6,Y6) #calculates the scores using the score_function f_regression of the features
        
        x = [i for i in range(len(fit.scores_))]
        x_best = [i for i in fit.get_support(indices=True)]
        scores = fit.scores_
        columns = np.array(df_features.columns[2:])
        
        
    elif featSelect_method == 'kBest-MI':
        features=SelectKBest(k=5,score_func=mutual_info_regression)
        fit=features.fit(X6,Y6)
        
        x = [i for i in range(len(fit.scores_))]
        x_best = [i for i in fit.get_support(indices=True)]
        scores = fit.scores_
        columns = np.array(df_features.columns[2:])
        

        
    elif featSelect_method == 'Forest-Regressor':
        model = RandomForestRegressor()
        model.fit(X6, Y6)
        
        
        scores = model.feature_importances_
        x = [i for i in range(len(scores))]
        y_best  = np.sort(scores)[-5:]
        x_best=[]
        for i in range(len(scores)):
            for y in y_best:
                if y==scores[i]:
                    x_best.append(i)
        columns = np.array(df_features.columns[2:])
        
    
    loading_output = None
    
    # Your plotting code here
    fig = None
    
    if featSelect_method:
        # Generate the plot based on the selected feature selection method
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=scores, name='All Features'))
        fig.add_trace(go.Bar(x=x_best, y=scores[x_best], name='Best Features'))

        # Update layout
        fig.update_layout(
            title='With ' + featSelect_method  + ' score',
            xaxis=dict(tickvals=x, ticktext=columns, tickangle=90),
            yaxis=dict(type='log'),
            barmode='overlay',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            width = 800,
            height = 600,
        )
    else:
        # Display the loading spinner while the graph is being updated
        loading_output = dbc.Spinner(
            color="primary",
            children="Loading...",
            style={"position": "absolute", "top": "50%", "left": "50%", "transform": "translate(100%, 100%)"},
            )
    
    return fig, loading_output
    

def RFRegressor(features,dftest):
    X_test = dftest[features]
    X_train = df_train[features]
    y_train = df_train['Power (kW)']
    parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 150, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 15,
              'max_leaf_nodes': None}
    RF_model = RandomForestRegressor(**parameters)
    RF_model.fit(X_train, y_train)
    y_pred_RF = RF_model.predict(X_test)
    return y_pred_RF

def LRegressor(features,dftest):
    X_test = dftest[features]
    X_train = df_train[features]
    y_train = df_train['Power (kW)']
    
    regr = linear_model.LinearRegression()
    regr.fit(X_train,y_train)
    y_pred_LR = regr.predict(X_test)
    return y_pred_LR

def DTRegressor(features,dftest):
    X_test = dftest[features]
    X_train = df_train[features]
    y_train = df_train['Power (kW)']

    DT_regr_model = DecisionTreeRegressor()
    DT_regr_model.fit(X_train, y_train)
    y_pred_DT = DT_regr_model.predict(X_test) 
    return y_pred_DT

def NNRegressor(features,dftest):
    X_test = dftest[features]
    X_train = df_train[features]
    y_train = df_train['Power (kW)']
    
    NN_model = MLPRegressor(hidden_layer_sizes=(5,10,5),max_iter=400)
    NN_model.fit(X_train,y_train)
    y_pred_NN = NN_model.predict(X_test)    
    
    return y_pred_NN

@app.callback([Output('models-features-plot', 'figure'), Output('table-container','children'),Output("loading-output-models", "children")],
              [Input('features-opt', 'value'),Input('model-opt','value'),
              Input('date-picker-range2', 'start_date'),  # Input component: start date
              Input('date-picker-range2', 'end_date')])

def update_fig7(features, models, start_date, end_date):
    loading_output = None
    fig = None
    
    if features and models:  # Check if features and models are selected
        fig = go.Figure()
        
        # Convert start_date and end_date strings to datetime objects
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter data based on selected start and end dates
        df_filtered = df_test[(df_test.index >= start_date) & (df_test.index <= end_date)]
        df_feat_filtered = df_features[(df_test.index >= start_date) & (df_test.index <= end_date)]
        #Y = df_filtered['Central (kWh)']
        
        MAEs,MBEs,MSEs,RMSEs,cvRMSEs, NMBEs = [],[],[],[],[],[]
        # Loop through selected models
        for model in models:
            if model == 'Linear Regression':
                y_pred = LRegressor(features, df_feat_filtered)
            elif model == 'Forest Regressor':
                y_pred = RFRegressor(features,df_feat_filtered)
            elif model =='Neural Networks':
                y_pred = NNRegressor(features,df_feat_filtered)
            elif model == 'Decision Tree Regressor':
                y_pred = DTRegressor(features,df_feat_filtered)
            MAE,MBE,MSE,RMSE,cvRMSE, NMBE = calcMetrics(df_filtered['Central (kWh)'],y_pred)
            MAEs.append(MAE)
            MBEs.append(MBE)
            MSEs.append(MSE)
            RMSEs.append(RMSE)
            cvRMSEs.append(cvRMSE)
            NMBEs.append(NMBE)
            
            # Add scatter plot for the predictions of the current model
            fig.add_trace(go.Scatter(
                x=df_filtered.index,  # Use filtered index
                y=y_pred[:len(df_filtered)],  # Use filtered length of y_pred
                mode='lines',
                name=model
            ))
        
        d = {'Methods': models, 'MAE': MAEs,'MBE': MBEs, 'MSE': MSEs, 'RMSE':RMSEs,'cvMSE':cvRMSEs ,'NMBE':NMBEs}
        df_metrics = pd.DataFrame(data=d)
        table = generate_table(df_metrics)
        
        # Add scatter plot for the real values
        fig.add_trace(go.Scatter(
            x=df_filtered.index,  # Use filtered index
            y=df_filtered['Central (kWh)'],  # Use filtered values of Y_real
            mode='lines',
            name='Real Values',
            line=dict(color='#2A2A2A')
        ))
        
        # Add layout properties
        fig.update_layout(
            title="Model Predictions",
            xaxis_title="Date",  # Update with appropriate x-axis label
            yaxis_title="Power (kW)",  # Update with appropriate y-axis label
        )
    else:
        # Display the loading spinner while the graph is being updated
        loading_output = dbc.Spinner(
            color="primary",
            children="Loading...",
            style={"position": "absolute", "top": "50%", "left": "50%", "transform": "translate(-50%, -50%)"},
        )
    
    return fig, table, loading_output

if __name__ == '__main__':
    app.run_server()
