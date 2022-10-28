import os

import servicefoundry.core as sfy
from servicefoundry import Build, PythonBuild, Resources, Service

from deploy import experiment_track
from model import train_model

def deploy_model():
    model, features, labels = train_model()
    experiment_track(model, features, labels)
    service = Service(
        name="trained-rfr-from-job",
        image=Build(
            build_spec=PythonBuild(
                command="python app.py",
            ),
        ),
        ports=[{"port": 8080}],
        resources=Resources(memory_limit=1500, memory_request=1000,
                            cpu_limit=1, cpu_request=0.5),
        env={"TFY_HOST": os.getenv('TFY_HOST'), "TFY_API_KEY": os.getenv(
            'TFY_API_KEY'), "WORKSPACE_FQN": os.getenv('WORKSPACE_FQN')}
    )
    service.deploy(workspace_fqn=os.getenv('WORKSPACE_FQN'))


def write_app():

    app = """import os
from datetime import datetime

import gradio as gr
import mlfoundry as mlf
import numpy as np
import pandas as pd

mlf_client = mlf.get_client()

runs = mlf_client.get_all_runs('train-job')

run = mlf_client.get_run(runs['run_id'][0])


model = mlf_client.get_model(
    f"model:truefoundry/arsh-anwar/train-job/Best_Model:{runs['run_name'][0].split('-')[3]}")
model = model.load()


df = run.get_dataset('features')

df = pd.DataFrame(df.features)

inputs = []
i = 0
sample = df.iloc[0:1].values.tolist()[0]
for x in df.columns:
    if df[x].dtype == 'object':
        inputs.append(gr.Textbox(label=x, value=sample[i]))
    elif df[x].dtype == 'float64' or df[x].dtype == 'int64':
        inputs.append(gr.Number(label=x, value=sample[i]),)
    i += 1


def predict(*val):
    print(val)
    global model
    if type(val) != list:
        val = [val]
    if type(val) != np.array:
        print('conv')
        val = np.array(val)
        print(val.shape)
    if val.ndim == 1:
        print('reshape')
        val = val.reshape(1, -1)
    pred = model.predict(val)
    return pred.tolist()[0]


app = gr.Interface(fn=predict, inputs=inputs,
                   outputs=gr.Textbox(label='Salary'), title='Salary Predictor', description=f'''## Predict the salary of a person based on their experience and education level
## Model Deployed at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}''')
app.launch(server_name="0.0.0.0", server_port=8080)
"""
    with open('app.py', 'w') as f:
        f.write(app)


if __name__ == "__main__":
    write_app()
    deploy_model()
