"""
This is a boilerplate pipeline 'data_preparation'
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
    return pipeline([node(func=nodes.drop_useless,
                          inputs=mapping(["test_data_final", "train_data_final"]),
                          outputs=None,
                          name="Drop useless colums")])
