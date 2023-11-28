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
                          outputs="loss",
                          name="loss_function"),
                     node(func=nodes.import_model,
                          inputs=["params:model_params.name",
                                  "params:model_params.llayer",
                                  "params:model_params.doratio"],
                          outputs=["tokenizer", "model"],
                          name="import_model"),
                     node(func=nodes.get_optimizer,
                          inputs=["model",
                                  "params:model_params.learningrate"],
                          outputs="optimizer",
                          name="optimizer"),
                     node(func=nodes.get_xla_device,
                          inputs=None,
                          outputs="device",
                          name="get_supported_device"),
                     node(func=nodes.loader,
                          inputs=["train_cleaned_fea",
                                  "tokenizer",
                                  "params:model_params.max_len",
                                  "params:train_params",
                                  "params:train_sample.datasize"],
                          outputs="trainloaded",
                          name="load_train_features"),
                     node(func=nodes.loader,
                          inputs=["test_cleaned_fea",
                                  "tokenizer",
                                  "params:model_params.max_len",
                                  "params:eval_params",
                                  "params:eval_sample.datasize"],
                          outputs="testloaded",
                          name="load_test_features"),
                     node(func=nodes.train_model,
                          inputs=mapping(["model",
                                          "loss",
                                          "optimizer",
                                          "params:model_params.epochs",
                                          "trainloaded",
                                          "device"]),
                          outputs=["bertmodel", "train_metrics"],
                          name="train_bert_model"),
                     node(func=nodes.eval_pt_model,
                          inputs=mapping(["bertmodel",
                                          "testloaded",
                                          "device"]),
                          outputs=["eval_metrics"],
                          name="evaluate_model")]
                    )
