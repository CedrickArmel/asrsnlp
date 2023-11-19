"""
This is a boilerplate pipeline 'train_bert'
generated using Kedro 0.18.14
"""
from kedro.pipeline import Pipeline, node, pipeline
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Returns the pipeline"""
    return pipeline([node(func=nodes.get_loss,
                          inputs=None,
                          outputs="loss",
                          name="Loss function"),
                     node(func=nodes.import_model,
                          inputs=["params:model_name", "lastlayer", "doratio"],
                          outputs=["tokenizer", "bertmodel"],
                          name="Import model"),
                     node(func=nodes.get_optimizer,
                          inputs=["bertmodel", "params:learningrate"],
                          outputs="optimizer",
                          name="Optmizer"),
                     node(func=nodes.get_device,
                          inputs=None,
                          outputs="device",
                          name="Get supporte device"),
                     node(),
                     node(),
                     node(),
                     node(),
                     node(),
                     node(),
                     node(),
                     node(),
                     node()]
                    )



"""            node(
func=set_columns_as_date,
inputs=['client'],
outputs='cl',
name="convert_date_columns_in_CL",
)
"""
