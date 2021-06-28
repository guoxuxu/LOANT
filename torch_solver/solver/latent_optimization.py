

def save_grad(var):
    def hook(grad):
        var.grad = grad
    return hook


def optimize_z(scaler, optimizer, epsilon, loss, ad_loss, z):
    optimizer.zero_grad()
    # LO: take second grad and compute updated loss
    handle = z.register_hook(save_grad(z))
    scaler.scale(loss).backward(retain_graph=True)
    delta_z = z.grad
    handle.remove()
    z = z - delta_z * epsilon
    return z
