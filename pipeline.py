from kfp import dsl
from kfp import components

def prepare_data_op():
    return dsl.ContainerOp(      
        name='Prepare Data',
        image='python:3.9',
        command=['python', 'dataset_prep.py'],
        file_outputs={
        'train': '/train.csv',
        'test': '/test.csv'
        }
    )

def train_model_op(train_data):
    return dsl.ContainerOp(
        name='Train Model',
        image='python:3.9',
        command=['python', 'train.py'],
        arguments=['--data', train_data],
        file_outputs={'model': '/model.pkl'}
    )

def evaluate_model_op(model, test_data):
    return dsl.ContainerOp(    
        name='Evaluate Model',
        image='python:3.9',
        command=['python', 'evaluate.py'],
        arguments=[
        '--model', model,
        '--test-data', test_data
        ],
        file_outputs={'metrics': '/metrics.json'}
    )

@dsl.pipeline(
    name='Diabetes Prediction Pipeline',
    description='Train and evaluate a diabetes prediction model'
)

def diabetes_pipeline():
    data_prep = prepare_data_op()
    train = train_model_op(data_prep.outputs['train'])
    evaluate = evaluate_model_op(
        train.outputs['model'],
        data_prep.outputs['test']
    )
# Compile pipeline
from kfp.compiler import Compiler
Compiler().compile(diabetes_pipeline, 'diabetes_pipeline.yaml')