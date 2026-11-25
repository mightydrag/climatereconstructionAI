import torch
import torch.nn.functional as F


def correlated_noise(shape, sigma_px, device, sigma_py=None):
    # spatially correlated unit-variance Gaussian field: white noise smoothed
    # with a separable Gaussian kernel, renormalized to unit std
    sigma_py = sigma_px if sigma_py is None else sigma_py
    z = torch.randn(shape, device=device)

    def kernel(sigma):
        ks = int(4 * sigma) | 1
        x = torch.arange(ks, device=device, dtype=z.dtype) - ks // 2
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        return g / g.sum(), ks

    gx, kx = kernel(sigma_px)
    gy, ky = kernel(sigma_py)
    z = F.conv2d(z, gx.view(1, 1, 1, -1), padding=(0, kx // 2))
    z = F.conv2d(z, gy.view(1, 1, -1, 1), padding=(ky // 2, 0))
    return z / z.std(dim=(-2, -1), keepdim=True)


def sample_hurdle(logit_p, k, mu, sigma_px=6.0, z_occ=None, z_amt=None):
    # one spatially coherent sample from the hurdle-gamma distribution via a
    # Gaussian copula; occurrence and amount use independent correlated fields
    # so they can have different decorrelation lengths; the gamma quantile
    # uses the Wilson-Hilferty approximation (torch has no Gamma.icdf),
    # accurate for k ≳ 0.5 (the head floors k at 0.5)
    normal = torch.distributions.Normal(0.0, 1.0)
    b, _, h, w = logit_p.shape
    if z_occ is None:
        z_occ = correlated_noise((b, 1, h, w), sigma_px, logit_p.device)
    if z_amt is None:
        z_amt = correlated_noise((b, 1, h, w), sigma_px, logit_p.device)
    wet = normal.cdf(z_occ) < torch.sigmoid(logit_p)
    z = z_amt
    base = (1.0 - 1.0 / (9.0 * k) + z / (3.0 * k.sqrt())).clamp_min(1e-3)
    amt = mu * base ** 3
    return wet * amt


def sample_members(logit_p, k, mu, n_members=1, sigma_px=6.0, coarse_pr=None, max_corr=10.0):
    # draw an ensemble of (n_members, B, C, H, W) fields; if coarse_pr is
    # given, each member is rescaled so its pooled field matches the coarse
    # field (per-member conservation, not just of the mean)
    members = [sample_hurdle(logit_p, k, mu, sigma_px=sigma_px) for _ in range(n_members)]
    out = torch.stack(members)  # (N, B, C, H, W)
    if coarse_pr is not None:
        n, b, c, h, w = out.shape
        flat = out.reshape(n * b, c, h, w)
        pooled = F.adaptive_avg_pool2d(flat, coarse_pr.shape[-2:])
        corr = (coarse_pr / pooled.clamp_min(1e-6)).clamp(1.0 / max_corr, max_corr)
        corr = F.interpolate(corr, size=(h, w), mode="bilinear", align_corners=False)
        out = out * corr.reshape(n, b, c, h, w)
    return out
