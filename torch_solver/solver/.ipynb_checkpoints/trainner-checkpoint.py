from torch.utils.data import DataLoader
import time
import torch
import numpy as np
import math


class Trainer(object):
    def __init__(self, model, optimizer, scheduler, train_set, train_logger, eval_solver, options, mode):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_set = train_set
        self.eval_solver = eval_solver
        self.options = options
        self.mode = mode
        self.train_logger = train_logger
        if math.isnan(len(train_set) / options['batch_size']):
            print('total step is NaN')
            raise TypeError
        else:
            self.total_steps = int(len(train_set) / options['batch_size']) * options['training_epochs']

    def run(self):
        model = self.model
        optimizer = self.optimizer
        model.train()

        if self.options['model_name'] in ['mtl', 'mtl_lo', 'mtl_maml', 'mtl_lo_ss', 'adda', 'adda2disc', 'adda_lo', 'adda2disc_lo', 'adda_lo_ss']:
            from .mtl_train_batch import train_batch
        elif self.options['model_name'] in ['bert', 'bert_lo']:
            from .bert_train_batch import train_batch
        else:
            raise TypeError

        multi_obj = self.options['multi_objective_lo']

        # Creates once at the beginning of training, mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        # train mtl save memory in Nvidia: 12345MiB -> 10389MiB, in memory reserved: 11247026176 -> 9235857408

        all_train_lines = []

        for epoch in range(0, self.options['training_epochs']):
            print(epoch)
            # shuffle take seeds
            train_loader = DataLoader(self.train_set, batch_size=self.options.get('batch_size'), shuffle=True, num_workers=4)  # shuffle determined by random seed
            start_step = epoch * len(train_loader)

            for i, batch in enumerate(train_loader):
                if (start_step + i) % self.options['log_every'] == 0:
                    log = True
                else:
                    log = False

                if self.options['model_name'] in ['adda', 'adda_lo', 'adda_lo_ss', 'adda2disc', 'adda2disc_lo']:
                    p = float(i + start_step) / self.total_steps
                    ad_weight = 2. / (1. + np.exp(-10 * p)) - 1
                    ad =True
                else:
                    ad_weight = 0
                    ad = False

                optimizer.zero_grad()
                # mixed precision training
                with torch.cuda.amp.autocast():
                    train_loss, batch_forward_time, train_log_line = train_batch(scaler, model, optimizer, batch, self.options, log=log, ad_weight=ad_weight, ad=ad, multi_obj=multi_obj)
                # Scales the loss, Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                time_start = time.perf_counter()
                scaler.scale(train_loss).backward()  # back propagation
                time_end = time.perf_counter()

                # unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
                self.scheduler.step()

                if (start_step + i) % self.options['log_every'] == 0:
                    # print to train file
                    time_elapsed = time_end - time_start
                    mem_alloc = torch.cuda.memory_allocated(self.options['cuda'])
                    mem_reserv = torch.cuda.memory_reserved(self.options['cuda'])
                    cur_lr = optimizer.param_groups[0]['lr']

                    if self.options['model_name'] in ['mtl_lo', 'mtl_maml', 'mtl_lo_ss', 'adda_lo', 'adda2disc_lo', 'adda_lo_ss']:
                        forward_time = ','.join([str(batch_forward_time[0]), str(batch_forward_time[1])])
                    else:
                        forward_time = str(batch_forward_time)

                    # print to train.txt
                    train_log_line = train_log_line + ','.join([forward_time, str(time_elapsed), str(mem_alloc), str(mem_reserv), str(cur_lr)])

                    if self.mode == 'COPft':
                        grad_norm_sh = [p.grad.norm().item() for n, p in model.named_parameters() if 'share_pooler.0.weight' in n and p.grad is not None]
                        weight_norm_sh = model.share_pooler[0].weight.norm().item()
                        if math.isnan(grad_norm_sh[0]):
                            print('grad_norm_sh[0] is NaN')
                            grad_norm_sh = 0
                        else:
                            grad_norm_sh = int(grad_norm_sh[0] * 10000) / 10000
                        if math.isnan(weight_norm_sh):
                            print('weight_norm_sh is NaN')
                            grad_norm_sh = 0
                        else:
                            weight_norm_sh = int(weight_norm_sh * 1000) / 10000
                        train_log_line = ','.join([train_log_line, str(grad_norm_sh), str(weight_norm_sh)])

                    all_train_lines += [train_log_line]

                    # evaluate dev for model selection
                    self.eval_solver.run_dev(self.model)
                    # evaluate test
                    self.eval_solver.run_test(self.model)
                #
                # if (start_step + i) > 200 and (start_step + i) % self.options['log_every'] == 0 and epoch < 4:
                #     path = os.path.join(self.options['model_save_path'], str(epoch)+str(i))
                #     ensure_dir(path)
                #     torch.save(model.state_dict(), os.path.join(path, 'model.pt'))

        for line in all_train_lines:
            self.train_logger.log(line)
        
        self.eval_solver.log_dev()
        self.eval_solver.log_test()