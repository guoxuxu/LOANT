# Author: Guo Xu (guoxu2025@gmail.com)
# Licence: Cite the paper (see README) whenever data/methods in this repo is used
# Date: 2020 April-OCT


def establish(model_name, LayerList, use_emoji, initialization):
    if model_name in ['bert', 'bert_lo', 'bert_mlp']:
        from .bert import BERT
        model = BERT(LayerList, model_name, use_emoji, initialization)
    elif model_name in ['ANT', 'LOANT', 'ANT_MAML']:
        from .ant import ANT
        model = ANT(LayerList, use_emoji, initialization)
    elif model_name in ['mtl', 'mtl_maml', 'mtl_lo']:
        from .mtl import MTL
        model = MTL(LayerList, use_emoji, initialization)
    else:
        raise TypeError
    return model