import yaml
import argparse
import torchmodels
from utils import str2bool, set_seed
import torch_loader
import torch_solver
import time
import os
import torch


def training(options, source, target, fold, mode):
    # setting decided, seed first!
    set_seed(options['seed'])

    # get prepared inputs data
    if source is None:
        train_set, dev_set, test_set = torch_loader.get_inputs(source=None, target=target, emoji=False, fold=fold, upsample=True)
    else:
        train_set, source_dev, source_test, target_dev, target_test = torch_loader.get_inputs(source=source, target=target, emoji=False, fold=fold, upsample=True)
        dev_set = [source_dev, target_dev]
        test_set = [source_test, target_test]

    # init model, (has randomness)
    model = torchmodels.get_model(model_name=options['model_name'], use_emoji=False, initialization=None, num_all_tokens=30647, num_added_tokens=125)

    if args.mode == 'STDft' and args.pretrained is True:
        model.load_state_dict(torch.load(options['pretrained_model_path']))

    # start training, (has randomness)
    filename = '_'.join(['lr', str(options['lr']), 'seed', str(options['seed'])])
    if options['model_name'] in ['mtl_lo', 'mtl_maml', 'LOANT']:
        filename += '_epsilon_' + str(options['epsilon'])

    if args.only_adver is True:
        filename += '_adonly'

    if source is None:
        dataset_name = target
    else:
        dataset_name = source + target

    if args.mode == 'STDft' and args.pretrained is True:
        filename = os.path.join(args.source, filename)

    options['path'] = os.path.join('new_results', dataset_name, options['model_name'], filename, fold)
    print(options['path'])
    options['model_save_path'] = os.path.join(dataset_name, options['model_name'], filename)

    torch_solver.run(
        model=model,
        train_set=train_set,
        dev_set=dev_set,
        test_set=test_set,
        options=options,
        mode=mode
    )


def tune(options, source, target, fold, tune_range, mode):
    time_start = time.perf_counter()

    for lr in tune_range:
        options['lr'] = lr
        print('source {} -> target {}, lr {}'.format(source, target, lr))
        training(options, source, target, fold=fold, mode=mode)  # training for this setting

    time_end = time.perf_counter()
    print('source {} -> target {}, time elapsed {}'.format(source, target, time_end - time_start))


if __name__ == '__main__':
    '''
    concerning random initialization affects, 1. determine data loader, 2. determine model.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='LOANT', help="mtl, mtl_maml, mtl_lo, ANT, LOANT, ant_maml")
    parser.add_argument("--seed", type=int, default=1, help="seed for initialization")
    parser.add_argument("--cuda", type=int, default=0, help="select devices to be visible")
    parser.add_argument("--mode", type=str, default='COPft', help="STDft, COPft")
    parser.add_argument("--pretrained", type=bool, default=True, help="if mode is STDft")
    parser.add_argument("--model_selected_every", type=int, default=200, help="")
    parser.add_argument("--only_adver", type=bool, default=False, help="")
    parser.add_argument("--source", type=str, default='Ptacek', help="Ptacek, Ghosh")
    parser.add_argument("--target", type=str, default='SemEval18', help="SemEval18, iSarcasm")
    args = parser.parse_args()

    args.pretrained = False
    args.weight_decay = False
    args.mixed_precision_training = True
    args.weight_decay = False
    args.mode = 'COPft'

    args.cfg = 'LOANT'
    with open(os.path.join(os.getcwd(), 'cfgs', args.cfg + '.yaml'), "r") as f:
        opts = yaml.safe_load(f)
    opts['seed'] = args.seed
    opts['cuda'] = args.cuda
    print('mode={}, config of {}, seed={}, cuda={}'.format(args.mode, args.cfg, args.seed, args.cuda))
    args.only_adver = True
    opts['epsilon'] = 1
    opts['only_adver'] = args.only_adver
    # opts['batch_size'] = 32

    tune_range = [4e-5]
    tune(opts, 'Ghosh', 'SemEval18', fold='fold-1', tune_range=tune_range, mode=args.mode)
