

def maml_optimize_w(scaler, model, optimizer, epsilon, loss):
    lr = optimizer.param_groups[0]['lr']
    optimizer.zero_grad()
    scaler.scale(loss).backward(retain_graph=False)

    for name, params in model.named_parameters():
        if 'bert' in name:
            if params is not None and params.grad is not None:
                params.data.copy_(params - epsilon * lr * params.grad)

    return model
