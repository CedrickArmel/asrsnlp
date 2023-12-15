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
                          outputs="bertloss",
                          name="loss_function_for_bert_trainning"),
                     node(func=nodes.import_model,
                          inputs=["params:bert_params.name",
                                  "params:bert_params.llayer",
                                  "params:bert_params.doratio"],
                          outputs=["berttokenizer", "bert"],
                          name="import_model_for_bert_trainning"),
                     node(func=nodes.get_optimizer,
                          inputs=["bert",
                                  "params:bert_params.learningrate"],
                          outputs="bertoptimizer",
                          name="optimizer_for_bert_trainning"),
                     node(func=nodes.get_device,
                          inputs=None,
                          outputs="bertdevice",
                          name="get_supported_device_for_bert_trainning"),
                     node(func=nodes.loader,
                          inputs=["train_cleaned_fea",
                                  "berttokenizer",
                                  "params:bert_params.max_len",
                                  "params:berttrain_params",
                                  "params:berttrain_sample.datasize"],
                          outputs="berttrainloaded",
                          name="load_train_features_for_bert_trainning"),
                     node(func=nodes.loader,
                          inputs=["test_cleaned_fea",
                                  "berttokenizer",
                                  "params:bert_params.max_len",
                                  "params:berteval_params",
                                  "params:berteval_sample.datasize"],
                          outputs="bertevalloaded",
                          name="load_test_features_for_bert_trainning"),
                     node(func=nodes.train_model,
                          inputs=mapping(["bert",
                                          "bertloss",
                                          "bertoptimizer",
                                          "params:bert_params.epochs",
                                          "berttrainloaded",
                                          "bertevalloaded",
                                          "bertdevice"]),
                          outputs=["bertmodel", "bert_metrics"],
                          name="train_bert_model")]
                    )