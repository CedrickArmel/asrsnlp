"""
This is a boilerplate pipeline 'train_bert'
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
                          outputs="bertloss_xla",
                          name="loss_function_for_bert_xla_trainning"),
                     node(func=nodes.import_model,
                          inputs=["params:bert_params.name",
                                  "params:bert_params.llayer",
                                  "params:bert_params.doratio"],
                          outputs=["berttokenizer_xla", "bertmodel_xla"],
                          name="import_model_for_bert_xla_trainning"),
                     node(func=nodes.get_optimizer,
                          inputs=["bertmodel_xla",
                                  "params:bert_params.learningrate"],
                          outputs="bertoptimizer_xla",
                          name="optimizer_for_bert_xla_trainning"),
                     node(func=nodes.get_xla_device,
                          inputs=None,
                          outputs="bertdevice_xla",
                          name="get_supported_device_for_bert_xla_trainning"),
                     node(func=nodes.loader,
                          inputs=["train_cleaned_fea",
                                  "berttokenizer_xla",
                                  "params:bert_params.max_len",
                                  "params:berttrain_params",
                                  "params:berttrain_sample.datasize"],
                          outputs="berttrainloaded_xla",
                          name="load_train_features_for_bert_xla_trainning"),
                     node(func=nodes.loader,
                          inputs=["test_cleaned_fea",
                                  "berttokenizer_xla",
                                  "params:bert_params.max_len",
                                  "params:berteval_params",
                                  "params:berteval_sample.datasize"],
                          outputs="bertevalloaded_xla",
                          name="load_test_features_for_bert_xla_trainning"),
                     node(func=nodes.train_model,
                          inputs=mapping(["bertmodel_xla",
                                          "bertloss_xla",
                                          "bertoptimizer_xla",
                                          "params:bert_params.epochs",
                                          "berttrainloaded_xla",
                                          "bertevalloaded_xla",
                                          "bertdevice_xla"]),
                          outputs=["bertmodel", "bert_metrics"],
                          name="train_bert_model_on_xla")]
                    )