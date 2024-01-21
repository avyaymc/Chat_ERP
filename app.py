from flask import Flask, render_template, jsonify, redirect, request
import config
from loader import get_answer
from dashboard import init_dashboard
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from dash import callback, Input, Output, State
import plotly.express as px
# from dashboard import create_dataframe
# df=create_dataframe

def page_not_found(e):
  return render_template('404.html'), 404

app = Flask(__name__,instance_relative_config=False)
dash_app=init_dashboard(app)
app.config.from_object(config.config['development'])

app.register_error_handler(404, page_not_found)

@app.route('/')
@app.route('/home/', methods = ['POST', 'GET'])
def index():
    if request.method == 'POST':
      prompt=request.form['prompt']
      res={}
      res['answer'] = get_answer(prompt)
      return jsonify(res),200
		
    return render_template('index.html', **locals())


@app.route('/dashboard/')
def render_dash():
    return redirect('/dash')



app = DispatcherMiddleware(app, {
    '/dash': dash_app.server
})



if __name__ == '__main__':
    run_simple('0.0.0.0', 8888,app, use_reloader=True, use_debugger=True)
