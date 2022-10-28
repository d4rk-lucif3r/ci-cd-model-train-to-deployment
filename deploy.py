import os

import mlfoundry as mlf
import servicefoundry.core as sfy
from servicefoundry import Build, Job, PythonBuild, Resources, Schedule

os.environ['WORKSPACE_FQN'] = '<your-workspace-fqn>'
os.environ['TFY_API_KEY'] = '<your-api-key>'
os.environ["TFY_HOST"] = "https://app.truefoundry.com/"


def experiment_track(model, features, labels):
    mlf_api = mlf.get_client()
    mlf_run = mlf_api.create_run(
        project_name='train-job', run_name='train-job-1')
    fn = mlf_run.log_model(name='Best_Model', model=model,
                           framework=mlf.ModelFramework.SKLEARN, description='My_Model')
    mlf_run.log_dataset("features", features)
    mlf_run.log_dataset("labels", labels)


def deploy_job():
    sfy.login(os.getenv('TFY_API_KEY'))
    image = Build(
        build_spec=PythonBuild(
            command="python service.py",
            requirements_path="requirements.txt",
        )
    )
    job = Job(
        name="rfr-train-and-publish-job",
        image=image,
        resources=Resources(memory_limit=1500, memory_request=1000,
                            cpu_limit=1, cpu_request=0.5),
        trigger=Schedule(
            schedule="0 9 * * *",
            concurrency_policy="Forbid"
        ),
        env={"TFY_HOST": os.getenv('TFY_HOST'), "TFY_API_KEY": os.getenv(
            'TFY_API_KEY'), "WORKSPACE_FQN": os.getenv('WORKSPACE_FQN')}
    )
    job.deploy(workspace_fqn=os.getenv('WORKSPACE_FQN'))

if __name__ == "__main__":
    deploy_job()
    print('Job deployed successfully')
