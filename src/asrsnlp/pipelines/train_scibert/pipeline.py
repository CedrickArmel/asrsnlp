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
                          outputs="scibertloss_xla",
                          name="loss_function_for_scibert_xla_trainning"),
                     node(func=nodes.import_model,
                          inputs=["params:scibert_params.name",
                                  "params:scibert_params.llayer",
                                  "params:scibert_params.doratio"],
                          outputs=["sciberttokenizer_xla", "scibertmodel_xla"],
                          name="import_model_for_scibert_xla_trainning"),
                     node(func=nodes.get_optimizer,
                          inputs=["scibertmodel_xla",
                                  "params:scibert_params.learningrate"],
                          outputs="scibertoptimizer_xla",
                          name="optimizer_for_scibert_xla_trainning"),
                     node(func=nodes.get_xla_device,
                          inputs=None,
                          outputs="scibertdevice_xla",
                          name="get_supported_device_for_scibert_xla_trainning"),
                     node(func=nodes.loader,
                          inputs=["train_cleaned_fea",
                                  "sciberttokenizer_xla",
                                  "params:scibert_params.max_len",
                                  "params:sciberttrain_params",
                                  "params:sciberttrain_sample.datasize"],
                          outputs="sciberttrainloaded_xla",
                          name="load_train_features_for_scibert_xla_trainning"),
                     node(func=nodes.loader,
                          inputs=["test_cleaned_fea",
                                  "sciberttokenizer_xla",
                                  "params:scibert_params.max_len",
                                  "params:sciberteval_params",
                                  "params:sciberteval_sample.datasize"],
                          outputs="scibertevalloaded_xla",
                          name="load_test_features_for_scibert_xla_trainning"),
                     node(func=nodes.train_model,
                          inputs=mapping(["scibertmodel_xla",
                                          "scibertloss_xla",
                                          "scibertoptimizer_xla",
                                          "params:scibert_params.epochs",
                                          "sciberttrainloaded_xla",
                                          "scibertevalloaded_xla",
                                          "scibertdevice_xla"]),
                          outputs=["scibertmodel", "scibert_metrics"],
                          name="train_scibert_model_on_xla")]
                    )
