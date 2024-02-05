import dash
import dash_table
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
from utils import get_type
import mysql.connector as connection
import pandas as pd

config = {
    'user': 'root',
    'password': 'Test123$',
    'host': 'localhost',
}

def create_dataframe():
    try:
        mydb = connection.connect(**config)
        query = "Select * from startups;"
        result_dataFrame = pd.read_sql(query,mydb)
        mydb.close() #close the connection
        return result_dataFrame
    except Exception as e:
        mydb.close()
        print(str(e))
        return "Exception occured"
    

def create_dataframe_csv():
    df=pd.read_csv('graph_data.csv')
    print(df.head())
    return df


html_layout = """
<!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
        </head>
        <body class="dash-template">
            <header>
              <div class="nav-wrapper">
                <a href="/">
                    <img src="/static/img/logo.png" class="logo" />
                    <h1>Plotly Dash Flask Tutorial</h1>
                  </a>
                <nav>
                </nav>
            </div>
            </header>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
"""

def create_data_table(df):
    """Create Dash datatable from Pandas DataFrame."""
    table = dash_table.DataTable(
        id="database-table",
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("records"),
        sort_action="native",
        sort_mode="native",
        page_size=300,
    )
    return table



def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    df2 = create_dataframe()
    app = dash.Dash(
        server=server,
        url_base_pathname='/dashboard/',
        external_stylesheets=[
            "/static/css/styles.css",
            "https://fonts.googleapis.com/css?family=Lato",
        ],
    )

    # Load DataFrame
    # df2 = create_dataframe()

    # Custom HTML layout
    app.index_string = html_layout

    app.layout = html.Div(
        children=[
            dcc.Graph(id='graph-with-slider'),
            dcc.Input(id='hidden-input', type='hidden', value=''),
            create_data_table(df2),
        ],
        id="dash-container",
    )
    
    @app.callback(Output('graph-with-slider', 'figure'), Input('hidden-input', 'value'), State('hidden-input', 'value'))
    def update_figure(n_clicks, hidden_value):
        df=create_dataframe_csv()
        type=get_type('type.txt')
        col1=df.columns[0]
        col2=df.columns[1]
        if type=='box':
            fig = px.box(df, y=col1)
        elif type=='pie':
            fig=px.pie(df, values=col2, names=col1)
        elif type=='hist':
            fig=px.histogram(x= df[col1])
        
        fig.update_layout(transition_duration=500)
        return fig

    return app

