"""
This is a boilerplate pipeline 'train_safeaero_gpu'
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
                          outputs="safeaeroloss_gpu",
                          name="loss_function_for_safeaero_gpu_trainning"),
                     node(func=nodes.import_model,
                          inputs=["params:safeaero_params.name",
                                  "params:safeaero_params.llayer",
                                  "params:safeaero_params.doratio"],
                          outputs=["safeaerotokenizer_gpu", "safeaeromodel_gpu"],
                          name="import_model_for_safeaero_gpu_trainning"),
                     node(func=nodes.get_optimizer,
                          inputs=["safeaeromodel_gpu",
                                  "params:safeaero_params.learningrate"],
                          outputs="safeaerooptimizer_gpu",
                          name="optimizer_for_safeaero_gpu_trainning"),
                     node(func=nodes.get_cuda_device,
                          inputs=None,
                          outputs="safeaerodevice_gpu",
                          name="get_supported_device_for_safeaero_gpu_trainning"),
                     node(func=nodes.loader,
                          inputs=["train_cleaned_fea",
                                  "safeaerotokenizer_gpu",
                                  "params:safeaero_params.max_len",
                                  "params:safeaerotrain_params",
                                  "params:safeaerotrain_sample.datasize"],
                          outputs="safeaerotrainloaded_gpu",
                          name="load_train_features_for_safeaero_gpu_trainning"),
                     node(func=nodes.loader,
                          inputs=["test_cleaned_fea",
                                  "safeaerotokenizer_gpu",
                                  "params:safeaero_params.max_len",
                                  "params:safeaeroeval_params",
                                  "params:safeaeroeval_sample.datasize"],
                          outputs="safeaeroevalloaded_gpu",
                          name="load_test_features_for_safeaero_gpu_trainning"),
                     node(func=nodes.train_model,
                          inputs=mapping(["safeaeromodel_gpu",
                                          "safeaeroloss_gpu",
                                          "safeaerooptimizer_gpu",
                                          "params:safeaero_params.epochs",
                                          "safeaerotrainloaded_gpu",
                                          "safeaeroevalloaded_gpu",
                                          "safeaerodevice_gpu"]),
                          outputs=["safeaeromodelgpu", "safeaero_metricsgpu"],
                          name="train_safeaero_model_on_gpu")]
                    )
