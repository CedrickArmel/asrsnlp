"""
This is a boilerplate pipeline 'train_scibert'
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
    """Returns the pipeline"""
    return pipeline([node(func=nodes.get_loss,
                          inputs=None,
                          outputs="scibertloss",
                          name="loss_function_for_scibert_trainning"),
                     node(func=nodes.import_model,
                          inputs=["params:scibert_params.name",
                                  "params:scibert_params.llayer",
                                  "params:scibert_params.doratio"],
                          outputs=["sciberttokenizer", "scibertmodel"],
                          name="import_model_for_scibert_trainning"),
                     node(func=nodes.get_optimizer,
                          inputs=["scibertmodel",
                                  "params:scibert_params.learningrate"],
                          outputs="scibertoptimizer",
                          name="optimizer_for_scibert_trainning"),
                     node(func=nodes.get_device,
                          inputs=None,
                          outputs="scibertdevice",
                          name="get_supported_device_for_scibert_trainning"),
                     node(func=nodes.loader,
                          inputs=["train_cleaned_fea",
                                  "sciberttokenizer",
                                  "params:scibert_params.max_len",
                                  "params:sciberttrain_params",
                                  "params:sciberttrain_sample.datasize"],
                          outputs="sciberttrainloaded",
                          name="load_train_features_for_scibert_trainning"),
                     node(func=nodes.loader,
                          inputs=["test_cleaned_fea",
                                  "sciberttokenizer",
                                  "params:scibert_params.max_len",
                                  "params:sciberteval_params",
                                  "params:sciberteval_sample.datasize"],
                          outputs="scibertevalloaded",
                          name="load_test_features_for_scibert_trainning"),
                     node(func=nodes.train_model,
                          inputs=mapping(["scibertmodel",
                                          "scibertloss",
                                          "scibertoptimizer",
                                          "params:scibert_params.epochs",
                                          "sciberttrainloaded",
                                          "scibertevalloaded",
                                          "scibertdevice"]),
                          outputs=["scibertmodel", "scibert_metrics"],
                          name="train_scibert_model")]
                    )
