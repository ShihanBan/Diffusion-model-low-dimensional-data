import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training import train_state
from typing import Any, Sequence, Tuple

# -----------------------------------
# Noise scheduler (beta schedule)
# -----------------------------------
def get_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> jnp.ndarray:
    return jnp.linspace(beta_start, beta_end, timesteps)

class NoiseSchedule:
    def __init__(self, betas: jnp.ndarray):
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = jnp.cumprod(self.alphas)
        self.sqrt_alpha_bars = jnp.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = jnp.sqrt(1.0 - self.alpha_bars)

    def q_sample(self, x0: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray) -> jnp.ndarray:
        a = self.sqrt_alpha_bars[t]
        b = self.sqrt_one_minus_alpha_bars[t]
        # reshape factors to broadcast
        view = [x0.shape[0]] + [1] * (x0.ndim - 1)
        a = a.reshape(view)
        b = b.reshape(view)
        return a * x0 + b * noise

# -----------------------------------
# U-Net Model for noise prediction
# -----------------------------------
class ResidualBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, temb):
        # Use GroupNorm with one group to handle arbitrary channels
        h = nn.GroupNorm(num_groups=1, name='gn1')(x)
        h = nn.swish(h)
        h = nn.Conv(self.features, kernel_size=(3,3), padding='SAME')(h)

        h += nn.Dense(self.features)(nn.swish(temb))[:, None, None, :]

        h = nn.GroupNorm(num_groups=1, name='gn2')(h)
        h = nn.swish(h)
        h = nn.Conv(self.features, kernel_size=(3,3), padding='SAME',
                    kernel_init=nn.initializers.zeros)(h)
        return x + h

class UNet(nn.Module):
    channels: Sequence[int]
    time_embed_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # Time embedding
        temb = nn.Dense(self.time_embed_dim)(t.astype(jnp.float32))
        temb = nn.swish(temb)
        temb = nn.Dense(self.time_embed_dim)(temb)

        hs = []
        h = x
        for feat in self.channels:
            h = ResidualBlock(feat)(h, temb)
            hs.append(h)
            h = nn.avg_pool(h, window_shape=(2,2), strides=(2,2))

        h = ResidualBlock(self.channels[-1] * 2)(h, temb)

        for feat, h_skip in zip(reversed(self.channels), reversed(hs)):
            h = nn.ConvTranspose(feat, kernel_size=(4,4), strides=(2,2), padding='SAME')(h)
            h = jnp.concatenate([h, h_skip], axis=-1)
            h = ResidualBlock(feat)(h, temb)

        h = nn.GroupNorm(num_groups=1)(h)
        h = nn.swish(h)
        return nn.Conv(x.shape[-1], kernel_size=(3,3), padding='SAME')(h)

# -----------------------------------
# Gaussian Diffusion class
# -----------------------------------
class GaussianDiffusion:
    def __init__(self,
                 model: nn.Module,
                 timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 lr: float = 1e-4):
        self.timesteps = timesteps
        betas = get_beta_schedule(timesteps, beta_start, beta_end)
        self.noise_schedule = NoiseSchedule(betas)

        rng = jax.random.PRNGKey(0)
        dummy_x = jnp.zeros((1, 32, 32, 3))
        dummy_t = jnp.zeros((1,), dtype=jnp.int32)
        params = model.init(rng, dummy_x, dummy_t)
        tx = optax.adamw(lr)
        self.state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx
        )

    @jax.jit
    def p_loss(self, params, x0, t, key):
        noise = jax.random.normal(key, x0.shape)
        x_noisy = self.noise_schedule.q_sample(x0, t, noise)
        pred_noise = self.state.apply_fn({'params': params}, x_noisy, t)
        return jnp.mean((noise - pred_noise) ** 2)

    @jax.jit
    def train_step(self, state, batch, key):
        def loss_fn(params):
            bsz = batch.shape[0]
            t = jax.random.randint(key, (bsz,), 0, self.timesteps)
            return self.p_loss(params, batch, t, key)

        grads = jax.grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads)

    def sample(self, rng: jax.random.KeyArray, shape: Tuple[int]):
        x = jax.random.normal(rng, shape)
        for i in reversed(range(self.timesteps)):
            t = jnp.full((shape[0],), i, dtype=jnp.int32)
            eps = self.state.apply_fn({'params': self.state.params}, x, t)
            alpha = self.noise_schedule.alphas[i]
            alpha_bar = self.noise_schedule.alpha_bars[i]
            beta = self.noise_schedule.betas[i]

            coef1 = 1 / jnp.sqrt(alpha)
            coef2 = beta / jnp.sqrt(1 - alpha_bar)
            mean = coef1 * (x - coef2 * eps)
            if i > 0:
                noise = jax.random.normal(rng, shape)
                x = mean + jnp.sqrt(beta) * noise
            else:
                x = mean
        return x

if __name__ == "__main__":
    channels = [64, 128, 256]
    model = UNet(channels=channels)
    diffusion = GaussianDiffusion(model)
    print("Initialized diffusion timesteps=", diffusion.timesteps)
