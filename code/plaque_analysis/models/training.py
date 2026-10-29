"""One inner training/validation fit with restoration of its best epoch."""
from copy import deepcopy
import numpy as np
import torch
from plaque_analysis.models.network import FTRegressor, optimizer_groups
from plaque_analysis.models.preprocessing import FeatureTransform, TargetTransform


def train_regressor(train_x, train_y, validation_x, validation_y, names, config, *, seed):
    seed = int(seed) % (2**32 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    transform = FeatureTransform("quantile", quantiles=config.quantiles, seed=seed).fit(train_x, names)
    target = TargetTransform().fit(train_y)
    device = torch.device(config.network.device)
    x = torch.as_tensor(transform.transform(train_x), dtype=torch.float32, device=device)
    y = torch.as_tensor(target.transform(train_y), dtype=torch.float32, device=device)
    vx = torch.as_tensor(transform.transform(validation_x), dtype=torch.float32, device=device)
    vy = torch.as_tensor(target.transform(validation_y), dtype=torch.float32, device=device)
    model = FTRegressor(x.shape[1], config.network).to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config.network.weight_decay),
                                 lr=config.network.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    best, stale, history, best_state = float("inf"), 0, [], None
    for epoch in range(config.network.max_epochs):
        model.train()
        order = torch.randperm(len(x), generator=generator).to(device)
        for indices in order.split(config.network.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x[indices]) - y[indices]) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; inspect input scales and settings")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(torch.mean((model(vx) - vy) ** 2).cpu())
        history.append({"epoch": epoch + 1, "validation_standardized_mse": score})
        if not np.isfinite(score):
            raise RuntimeError("Non-finite validation loss")
        if score < best:
            best, stale, best_epoch = score, 0, epoch + 1
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.network.patience:
                break
    model.load_state_dict(best_state)
    audit = {**transform.audit(), "target_mean": target.mean_, "target_scale": target.scale_,
             "seed": seed, "best_epoch": best_epoch, "epochs_run": len(history),
             "best_validation_mse": best, "history": history}
    return model, transform, target, audit


def predict_regressor(fitted, values, batch_size=1024):
    model, transform, target, _ = fitted
    device = next(model.parameters()).device
    values = transform.transform(values)
    pieces = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            x = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device=device)
            pieces.append(model(x).cpu().numpy())
    return target.inverse_transform(np.concatenate(pieces))
