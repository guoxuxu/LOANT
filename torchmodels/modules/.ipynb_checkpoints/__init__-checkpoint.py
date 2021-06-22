# Author: Guo Xu (guoxu2025@gmail.com)
# Licence: Cite the paper (see README) whenever data/methods in this repo is used
# Date: 2020 April-OCT


def establish(model_name, LayerList, use_emoji, initialization):
    if model_name in ['bert', 'bert_lo', 'bert_mlp']:
        from .bert import BERT
        model = BERT(LayerList, model_name, use_emoji, initialization)
    elif model_name in ['adda', 'adda_lo', 'adda_lo_ss']:
        from .adda import ADDA
        model = ADDA(LayerList, use_emoji, initialization)
    elif model_name in ['adda2disc', 'adda2disc_lo']:
        from .adda2disc import ADDA2DISC
        model = ADDA2DISC(LayerList, use_emoji, initialization)
    elif model_name in ['mtl', 'mtl_maml', 'mtl_lo', 'mtl_lo_ss']:
        from .mtl import MTL
        model = MTL(LayerList, use_emoji, initialization)
    elif model_name == 'mtl_sep':
        from .mtl_sep import MTL_SEP
        model = MTL_SEP(LayerList, use_emoji, initialization)
    else:
        raise TypeError
    return model