"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([node(func=nodes.dropna_row,
                          inputs=["train", "params:sub"],
                          outputs="train_cleaned",
                          name="Clean train data"),
                     node(func=nodes.dropna_row,
                          inputs=["test", "params:sub"],
                          outputs="test_cleaned",
                          name="Clean test data"),
                     node(func=nodes.target_encoder,
                          inputs=["train_cleaned", "params:target", "params:labels"],
                          outputs=None,
                          name="Encode train multilabet target"),
                     node(func=nodes.target_encoder,
                          inputs=["test_cleaned", "params:target", "params:labels"],
                          outputs=None,
                          name="Encode test multilabet target")])
