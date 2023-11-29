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
                          outputs="safeaeroloss_xla",
                          name="loss_function_for_safeaero_xla_trainning"),
                     node(func=nodes.import_model,
                          inputs=["params:safeaero_params.name",
                                  "params:safeaero_params.llayer",
                                  "params:safeaero_params.doratio"],
                          outputs=["safeaerotokenizer_xla", "safeaeromodel_xla"],
                          name="import_model_for_safeaero_xla_trainning"),
                     node(func=nodes.get_optimizer,
                          inputs=["safeaeromodel_xla",
                                  "params:safeaero_params.learningrate"],
                          outputs="safeaerooptimizer_xla",
                          name="optimizer_for_safeaero_xla_trainning"),
                     node(func=nodes.get_cuda_device,
                          inputs=None,
                          outputs="safeaerodevice_xla",
                          name="get_supported_device_for_safeaero_xla_trainning"),
                     node(func=nodes.loader,
                          inputs=["train_cleaned_fea",
                                  "safeaerotokenizer_xla",
                                  "params:safeaero_params.max_len",
                                  "params:safeaerotrain_params",
                                  "params:safeaerotrain_sample.datasize"],
                          outputs="safeaerotrainloaded_xla",
                          name="load_train_features_for_safeaero_xla_trainning"),
                     node(func=nodes.loader,
                          inputs=["test_cleaned_fea",
                                  "safeaerotokenizer_xla",
                                  "params:safeaero_params.max_len",
                                  "params:safeaeroeval_params",
                                  "params:safeaeroeval_sample.datasize"],
                          outputs="safeaeroevalloaded_xla",
                          name="load_test_features_for_safeaero_xla_trainning"),
                     node(func=nodes.train_model,
                          inputs=mapping(["safeaeromodel_xla",
                                          "safeaeroloss_xla",
                                          "safeaerooptimizer_xla",
                                          "params:safeaero_params.epochs",
                                          "safeaerotrainloaded_xla",
                                          "safeaeroevalloaded_xla",
                                          "safeaerodevice_xla"]),
                          outputs=["safeaeromodel", "safeaero_metrics"],
                          name="train_safeaero_model_on_xla")]
                    )