"""
This is a boilerplate pipeline 'train_bert_gpu'
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
                          outputs="bertloss_gpu",
                          name="loss_function_for_bert_gpu_trainning"),
                     node(func=nodes.import_model,
                          inputs=["params:bert_params.name",
                                  "params:bert_params.llayer",
                                  "params:bert_params.doratio"],
                          outputs=["berttokenizer_gpu", "bertmodel_gpu"],
                          name="import_model_for_bert_gpu_trainning"),
                     node(func=nodes.get_optimizer,
                          inputs=["bertmodel_gpu",
                                  "params:bert_params.learningrate"],
                          outputs="bertoptimizer_gpu",
                          name="optimizer_for_bert_gpu_trainning"),
                     node(func=nodes.get_cuda_device,
                          inputs=None,
                          outputs="bertdevice_gpu",
                          name="get_supported_device_for_bert_gpu_trainning"),
                     node(func=nodes.loader,
                          inputs=["train_cleaned_fea",
                                  "berttokenizer_gpu",
                                  "params:bert_params.max_len",
                                  "params:berttrain_params",
                                  "params:berttrain_sample.datasize"],
                          outputs="berttrainloaded_gpu",
                          name="load_train_features_for_bert_gpu_trainning"),
                     node(func=nodes.loader,
                          inputs=["test_cleaned_fea",
                                  "berttokenizer_gpu",
                                  "params:bert_params.max_len",
                                  "params:berteval_params",
                                  "params:berteval_sample.datasize"],
                          outputs="bertevalloaded_gpu",
                          name="load_test_features_for_bert_gpu_trainning"),
                     node(func=nodes.train_model,
                          inputs=mapping(["bertmodel_gpu",
                                          "bertloss_gpu",
                                          "bertoptimizer_gpu",
                                          "params:bert_params.epochs",
                                          "berttrainloaded_gpu",
                                          "bertevalloaded_gpu",
                                          "bertdevice_gpu"]),
                          outputs=["bertmodelgpu", "bert_metricsgpu"],
                          name="train_bert_model_on_gpu")]
                    )