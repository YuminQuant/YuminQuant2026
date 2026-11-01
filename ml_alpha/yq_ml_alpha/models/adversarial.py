"""Sample-independent WGAN-GP losses and isolated generator updates."""
from contextlib import contextmanager


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
