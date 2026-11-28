"""Training-time input perturbations and WGAN-GP helpers."""
from contextlib import contextmanager


def linf_adversarial_inputs(torch, inputs, predict, target, *, steps, epsilon, step_size):
    """Projected gradient ascent in channel-standard-deviation units."""
    inputs = inputs.detach()
    scale = inputs.std(dim=(0, 1), unbiased=False, keepdim=True)
    budget, step = epsilon * scale, step_size * scale
    perturbed = inputs
    for _ in range(steps):
        perturbed = perturbed.detach().requires_grad_(True)
        mse = (predict(perturbed) - target).square().mean()
        (gradient,) = torch.autograd.grad(mse, perturbed)
        with torch.no_grad():
            delta = (perturbed - inputs + step * gradient.sign()).clamp(min=-budget, max=budget)
            perturbed = inputs + delta
    return perturbed


@contextmanager
def batchnorm_without_running_updates(*layers):
    """Use training batch statistics while leaving the running buffers untouched."""
    tracking = [layer.track_running_stats for layer in layers]
    try:
        for layer in layers:
            layer.track_running_stats = False
        yield
    finally:
        for layer, enabled in zip(layers, tracking):
            layer.track_running_stats = enabled


def noisy_samples(torch, values, std):
    return values if not std else values + std * torch.randn_like(values)


def gradient_penalty(torch, critic, real, fake):
    if real.shape != fake.shape or real.ndim != 2 or len(real) < 2:
        raise ValueError("WGAN-GP requires equally sized real/fake matrices with at least two rows")
    fraction = torch.rand((len(real), 1), device=real.device, dtype=real.dtype)
    mixed = (fraction * real.detach() + (1 - fraction) * fake.detach()).requires_grad_(True)
    gradient = torch.autograd.grad(critic(mixed).sum(), mixed, create_graph=True)[0]
    return (gradient.flatten(1).norm(dim=1) - 1).square().mean()


def critic_objective(torch, critic, real, fake, *, gp_weight):
    real, fake = real.detach(), fake.detach()
    real_score, fake_score = critic(real).mean(), critic(fake).mean()
    penalty = gradient_penalty(torch, critic, real, fake)
    return fake_score - real_score + gp_weight * penalty, real_score - fake_score, penalty


@contextmanager
def frozen_critics(critics):
    states = [(critic, critic.training, [(p, p.requires_grad) for p in critic.parameters()]) for critic in critics]
    try:
        for critic, _, parameters in states:
            critic.eval()
            for parameter, _ in parameters:
                parameter.requires_grad_(False)
        yield
    finally:
        for critic, training, parameters in states:
            critic.train(training)
            for parameter, requires_grad in parameters:
                parameter.requires_grad_(requires_grad)
