import torch

def noise_scheduler(beta_min = 1e-4, beta_max = 0.02, T = 500):

    print("Adopting a linear scheduler")

    
    ts = torch.arange(T, dtype=torch.float32)
    betas = beta_min +  ts/(T-1) * (beta_max - beta_min)

    
    alphas = 1.0 - betas
    alpha_hats = torch.cumprod(alphas, dim = 0)
    alpha_hats_sqrt = torch.sqrt(alpha_hats)
    one_minus_alpha_hats_sqrt = torch.sqrt((1.0 - alpha_hats))


    alpha_hats_prev = torch.cat([
        torch.ones(1, dtype=alpha_hats.dtype),
        alpha_hats[:-1]
    ])

    posterior_variance = (
        betas * (1.0 - alpha_hats_prev) / (1.0 - alpha_hats)
    )

    
    return {
    "betas": betas,
    "alphas": alphas,
    "alpha_hats": alpha_hats,
    "alpha_hats_sqrt": alpha_hats_sqrt,
    "one_minus_alpha_hats_sqrt": one_minus_alpha_hats_sqrt,
    "posterior_variance": posterior_variance,
    }


def add_noise_onestep(x_prev, t, scheduler_dict):
    added_noise = torch.randn_like(x_prev)
    x_current = scheduler_dict['alphas'][t].sqrt() * x_prev + scheduler_dict['betas'][t].sqrt() * added_noise

    return x_current





def add_noise_onestep_cum_sampler(x_0, t, scheduler_dict, no_samples = 1000):
    output = []
    for _ in range(no_samples):
        x = x_0
        for i in range(t + 1):
            x = add_noise_onestep(x, i, scheduler_dict)

        output.append(x)

    return torch.cat(output, dim = 0)



def add_noise_from_zero(x0, t, noise_dict, device = 'cpu'):
    added_noise = torch.randn_like(x0)
    xt = x0 * noise_dict["alpha_hats_sqrt"][t].view(-1,1) + added_noise *  noise_dict["one_minus_alpha_hats_sqrt"][t].view(-1, 1)

    return xt.to(device), added_noise.to(device)



@torch.no_grad()
def ddpm_sampler(sample_dim, n_sample, noise_dict, model, cond = None, t_steps = 500, device = 'cpu', everystep = False):
    x_t = torch.randn(n_sample,1,sample_dim).to(device)
    
    if everystep:
        x_per_frame = []
        x_per_frame.append(x_t.detach().cpu().clone())
        x_0_condition_t = []
        
    for i in reversed(range(t_steps)):
        t = torch.full((n_sample,), i)
        t = t.to(device)
        pred_noise = model(x_t, t, cond)

        if everystep:
            temp_x_0_condion_t = (x_t - noise_dict['one_minus_alpha_hats_sqrt'][i] * pred_noise)/noise_dict['alpha_hats_sqrt'][i]

        if i > 0:
            post_var = (1 - noise_dict['alpha_hats'][i-1])/(1 - noise_dict['alpha_hats'][i]) * noise_dict['betas'][i]

        mean_t = 1/noise_dict['alphas'][i].sqrt() * (x_t - (1 - noise_dict['alphas'][i])/(noise_dict['one_minus_alpha_hats_sqrt'][i]) * pred_noise)
        step_wise_noise = torch.randn(n_sample,1,sample_dim).to(device)


        if i > 0:
            x_t = mean_t + post_var.sqrt() * step_wise_noise
        else:
            x_t = mean_t

        if everystep:
            x_per_frame.append(x_t.detach().cpu().clone())
            x_0_condition_t.append(temp_x_0_condion_t.detach().cpu().clone())

    
    if everystep:
        return x_t, x_per_frame, x_0_condition_t
        
    else:
        return x_t
        