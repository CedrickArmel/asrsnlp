"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline
from . import nodes


def mapping(inputs: list[str]) -> dict[str, str]:
    """Helper function that creates a mapping value-value\n
    from value of the list for strings.

    Args:
        inputs (list): _Input list_

    Returns:
        dict: _The mapping_
    """
    return {k: k for k in inputs}


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([node(func=nodes.dropna_row,
                          inputs=["train_data_prm", "params:sub"],
                          outputs="train_cleaned",
                          name="clean_train_data"),
                     node(func=nodes.dropna_row,
                          inputs=["test_data_prm", "params:sub"],
                          outputs="test_cleaned",
                          name="clean_test_data"),
                     node(func=nodes.target_encoder,
                          inputs=mapping(["train_cleaned",
                                          "test_cleaned",
                                          "params:target",
                                          "params:labels"]),
                          outputs=["train_cleaned_fea", "test_cleaned_fea"],
                          name="encode_data_multilabet_target")])
