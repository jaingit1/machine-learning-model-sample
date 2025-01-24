import kfp
from kfp import dsl
from kfp.components import load_component_from_text

# Define common environment variables
GIT_REPO_URL = "https://github.com/jaingit1/machine-learning-model-sample"
GIT_BRANCH = "master"

# Define your components using YAML descriptions
prepare_data = load_component_from_text(f"""
name: Prepare Data
outputs:
- {{name: train, type: String}}
- {{name: test, type: String}}
implementation:
  container:
    image: jaindocker1/python:3.9-git
    command:
    - sh
    - -c
    - |
      git clone --branch {GIT_BRANCH} {GIT_REPO_URL} && 
      python machine-learning-model-sample/dataset_prep.py &&
      mv train.csv /tmp/kfp/outputs/train &&
      mv test.csv /tmp/kfp/outputs/test
    fileOutputs:
      train: /tmp/kfp/outputs/train
      test: /tmp/kfp/outputs/test
""")

train_model = load_component_from_text(f"""
name: Train Model
inputs:
- {{name: train_data, type: String}}
outputs:
- {{name: model, type: String}}
implementation:
  container:
    image: jaindocker1/python:3.9-git
    command:
    - sh
    - -c
    - |
      git clone --branch {GIT_BRANCH} {GIT_REPO_URL} && 
      python machine-learning-model-sample/train.py --data {{train_data}} &&
      mv model.pkl /tmp/kfp/outputs/model
    fileOutputs:
      model: /tmp/kfp/outputs/model
""")

evaluate_model = load_component_from_text(f"""
name: Evaluate Model
inputs:
- {{name: model, type: String}}
- {{name: test_data, type: String}}
outputs:
- {{name: metrics, type: String}}
implementation:
  container:
    image: jaindocker1/python:3.9-git
    command:
    - sh
    - -c
    - |
      git clone --branch {GIT_BRANCH} {GIT_REPO_URL} && 
      python machine-learning-model-sample/evaluate.py --model {{model}} --test-data {{test_data}} &&
      mv metrics.json /tmp/kfp/outputs/metrics
    fileOutputs:
      metrics: /tmp/kfp/outputs/metrics
""")

@dsl.pipeline(
    name='Diabetes Prediction Pipeline',
    description='Train and evaluate a diabetes prediction model'
)
def diabetes_pipeline():
    data_prep = prepare_data()
    train = train_model(train_data=data_prep.outputs['train'])
    evaluate = evaluate_model(
        model=train.outputs['model'],
        test_data=data_prep.outputs['test']
    )

# Compile pipeline
from kfp.compiler import Compiler
Compiler().compile(diabetes_pipeline, 'diabetes_pipeline_3.yaml')
