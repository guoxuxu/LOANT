__version__ = "0.0.1"

from . import utils
from .common import _init_layers
from . import modules


'''
General Encoder + Classifier -> Targeted Task-specific fine-tuning (bert)
General Encoder + Classifier -> Task-specific fine-tuned -> Targeted Task-specific fine-tuning
General Encoder + Source Classifier + Target Classifier -> Multi-Task fined-tuned
General Encoder + Source Classifier + Target Classifier + Domain Classifier -> Multi-Task fined-tuned with ADDA (DATNet)
General Encoder + Source Classifier + Target Classifier + Domain Classifier + Provate Domain Classifiers -> Multi-Task fined-tuned with ADDA (Style data paper)

'''
# # don't name the folder as 'models'


def get_tokenizer(add_new_tokens:bool):
    tokenizer, num_added_tokens = utils.init_bert_tokenizer(add_new_tokens=add_new_tokens)
    return tokenizer, num_added_tokens


def get_model(model_name, use_emoji, initialization, num_all_tokens, num_added_tokens):
    LayerList = _init_layers(num_all_tokens, num_added_tokens)
    model = modules.establish(model_name, LayerList, use_emoji, initialization)
    return model