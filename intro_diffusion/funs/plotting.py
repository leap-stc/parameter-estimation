import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation


def plot_batched_lines(lines, labels=None, x=None, max_lines=16, figsize=(8, 3)):
    """
    lines:  Tensor or array with shape [batch_size, line_length]
    labels: Tensor/list with shape [batch_size], optional
    x:      Tensor/array with shape [line_length], optional
    """

    if isinstance(lines, torch.Tensor):
        lines = lines.detach().cpu()

    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()

    n_lines = min(lines.shape[0], max_lines)

    if x is None:
        x = torch.arange(lines.shape[1])

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()

    plt.figure(figsize=figsize)

    for i in range(n_lines):
        if labels is None:
            label = f"sample {i}"
        else:
            label = f"sample {i}, label={labels[i].item():.2f}"

        plt.plot(x, lines[i], label=label)

    plt.xlabel("")
    plt.ylabel("")
    plt.legend()
    plt.tight_layout()
    plt.show()



from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.pyplot as plt

def animate_forward_noise(
    image,
    add_noise_onestep,
    scheduler_dict,
    T=500,
    interval=50,
    step=5
):
    x = image.clone()

    # Handle batched or unbatched input
    if x.ndim == 1:
        original_y = x.detach().cpu().numpy()
    else:
        original_y = x.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 3))

    # Fixed original line
    
    # Updating noisy line
    noisy_line, = ax.plot(
        original_y,
        color="navy",
        alpha=0.8,
        linewidth = 1.0,
    )

    ax.plot(
        original_y,
        color="red",
        linewidth=1.5,
    )


    ax.set_ylim(-2, 2)
    ax.legend()
    ax.set_title("Forward diffusion step: t = 0")

    def update(frame):
        nonlocal x

        target_t = frame * step

        while update.current_t < target_t:
            x = add_noise_onestep(x, update.current_t, scheduler_dict)
            update.current_t += 1

        if x.ndim == 1:
            noisy_y = x.detach().cpu().numpy()
        else:
            noisy_y = x.detach().cpu().numpy()

        noisy_line.set_ydata(noisy_y)
        ax.set_title(f"Forward diffusion step: t = {target_t}")

        return noisy_line,

    update.current_t = 0

    anim = FuncAnimation(
        fig,
        update,
        frames=T // step + 1,
        interval=interval,
        blit=False
    )

    plt.close(fig)
    return anim



import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def animate_reverse_process(x_per_frame, figsize=(4, 2), ylim=None):
    """
    Simple reverse diffusion animation.

    Input:
        x_per_frame: list of tensors from sampler (typically each [B, 1, L]).
        figsize: matplotlib figure size tuple, e.g. (10, 4).
        ylim: optional tuple (ymin, ymax). If None, limits are auto-computed.

    Output:
        matplotlib.animation.FuncAnimation
    """
    if not isinstance(x_per_frame, list) or len(x_per_frame) == 0:
        raise ValueError("x_per_frame must be a non-empty list.")

    frames = []
    for x in x_per_frame:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
        else:
            x = torch.as_tensor(x)

        if x.ndim == 3:
            # [B, 1, L] -> [L] using the first sample in batch
            x = x[0, 0, :]
        elif x.ndim == 2:
            # [B, L] -> [L] using the first sample in batch
            x = x[0, :]
        elif x.ndim != 1:
            raise ValueError("Each frame must be shaped like [B, 1, L], [B, L], or [L].")

        frames.append(x)

    frames = torch.stack(frames, dim=0).numpy()

    # Long animations rendered via to_jshtml can be truncated by the default embed limit.
    # Raise it so all reverse steps (including the final estimate) are included.
    plt.rcParams["animation.embed_limit"] = max(plt.rcParams.get("animation.embed_limit", 20.0), 100.0)

    fig, ax = plt.subplots(figsize=figsize)
    x_axis = torch.arange(frames.shape[-1])
    line, = ax.plot(x_axis, frames[0], color="tab:blue", linewidth=1.8)

    y_min = frames.min().item()
    y_max = frames.max().item()
    pad = max(1e-3, 0.1 * (y_max - y_min))

    ax.set_xlim(0, frames.shape[-1] - 1)
    if ylim is None:
        ax.set_ylim(y_min - pad, y_max + pad)
    else:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel("signal index")
    ax.set_ylabel("value")
    ax.set_title(f"Reverse diffusion frame 0/{len(frames) - 1}")

    def update(frame_idx):
        line.set_ydata(frames[frame_idx])
        ax.set_title(f"Reverse diffusion frame {frame_idx}/{len(frames) - 1}")
        return (line,)

    # frames=len(frames) ensures the final estimate (last element) is shown last.
    anim = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False, cache_frame_data=False)

    plt.close(fig)
    return anim
