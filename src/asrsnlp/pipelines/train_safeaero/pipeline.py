"""
This is a boilerplate pipeline 'train_safeaero'
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
                          outputs="safeaeroloss",
                          name="loss_function_for_safeaero_trainning"),
                     node(func=nodes.import_model,
                          inputs=["params:safeaero_params.name",
                                  "params:safeaero_params.llayer",
                                  "params:safeaero_params.doratio"],
                          outputs=["safeaerotokenizer", "safeaero"],
                          name="import_model_for_safeaero_trainning"),
                     node(func=nodes.get_optimizer,
                          inputs=["safeaero",
                                  "params:safeaero_params.learningrate"],
                          outputs="safeaerooptimizer",
                          name="optimizer_for_safeaero_trainning"),
                     node(func=nodes.get_device,
                          inputs=None,
                          outputs="safeaerodevice",
                          name="get_supported_device_for_safeaero_trainning"),
                     node(func=nodes.loader,
                          inputs=["train_cleaned_fea",
                                  "safeaerotokenizer",
                                  "params:safeaero_params.max_len",
                                  "params:safeaerotrain_params",
                                  "params:safeaerotrain_sample.datasize"],
                          outputs="safeaerotrainloaded",
                          name="load_train_features_for_safeaero_trainning"),
                     node(func=nodes.loader,
                          inputs=["test_cleaned_fea",
                                  "safeaerotokenizer",
                                  "params:safeaero_params.max_len",
                                  "params:safeaeroeval_params",
                                  "params:safeaeroeval_sample.datasize"],
                          outputs="safeaeroevalloaded",
                          name="load_test_features_for_safeaero_trainning"),
                     node(func=nodes.train_model,
                          inputs=mapping(["safeaero",
                                          "safeaeroloss",
                                          "safeaerooptimizer",
                                          "params:safeaero_params.epochs",
                                          "safeaerotrainloaded",
                                          "safeaeroevalloaded",
                                          "safeaerodevice"]),
                          outputs=["safeaeromodel", "safeaero_metrics"],
                          name="train_safeaero_model")]
                    )