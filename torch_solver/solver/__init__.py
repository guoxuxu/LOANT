

def train(model, optimizer, scheduler, train_set, train_logger, eval_solver, options, mode):
    from .trainner import Trainer
    handler = Trainer(model, optimizer, scheduler, train_set, train_logger, eval_solver, options, mode)
    handler.run()


def evaluate(dev_loader, test_loader, dev_logger, test_logger, eyeball_logger, cuda, model_name, mode:str):
    from .evaluater import Evaluater
    handler = Evaluater(dev_loader, test_loader, dev_logger, test_logger, eyeball_logger, cuda, model_name, mode)
    return handler


def get_metrics(TP, TN, FP, FN, P, N):
    from .metric import Metrics
    metric_handler = Metrics(TP, TN, FP, FN, P, N)
    return metric_handler.compute()