import kfp
from kfp import dsl
from kfp.components import load_component_from_text

# Define your components using YAML descriptions
prepare_data = load_component_from_text("""
name: Prepare Data
outputs:
- {name: train, type: String}
- {name: test, type: String}
implementation:
  container:
    image: python:3.9
    command:
    - python
    - dataset_prep.py
""")


train_model = load_component_from_text("""
name: Train Model
inputs:
- {name: train_data, type: String}
outputs:
- {name: model, type: String}
implementation:
  container:
    image: python:3.9
    command:
    - python
    - train.py
    args:
    - --data
    - {inputValue: train_data}
""")


evaluate_model = load_component_from_text("""
name: Evaluate Model
inputs:
- {name: model, type: String}
- {name: test_data, type: String}
outputs:
- {name: metrics, type: String}
implementation:
  container:
    image: python:3.9
    command:
    - python
    - evaluate.py
    args:
    - --model
    - {inputValue: model}
    - --test-data
    - {inputValue: test_data}
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
Compiler().compile(diabetes_pipeline, 'diabetes_pipeline.yaml')
