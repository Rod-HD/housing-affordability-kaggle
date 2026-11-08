"""MLP regressor used in the ensemble: quantile-normal inputs + missing flags, Huber loss, sample weights."""
import numpy as np, torch
from sklearn.preprocessing import QuantileTransformer

torch.set_num_threads(8)


def prep(Xt, Xv):
    med = Xt.median()
    miss_cols = [c for c in Xt.columns if Xt[c].isna().mean() > 0.01]
    At = Xt.fillna(med); Av = Xv.fillna(med)
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=10**6, random_state=0)
    Zt = qt.fit_transform(At); Zv = qt.transform(Av)
    Mt = Xt[miss_cols].isna().values.astype("float32"); Mv = Xv[miss_cols].isna().values.astype("float32")
    return np.hstack([Zt, Mt]).astype("float32"), np.hstack([Zv, Mv]).astype("float32")


def mlp_fit_predict(Zt, yt, wt, Zv, seed, epochs=160):
    torch.manual_seed(seed); np.random.seed(seed)
    d = Zt.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.SiLU(), torch.nn.Dropout(0.15),
                              torch.nn.Linear(256, 128), torch.nn.SiLU(), torch.nn.Dropout(0.10),
                              torch.nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * int(np.ceil(len(yt) / 128)))
    X = torch.tensor(Zt); Y = torch.tensor(((yt - 50) / 30).astype("float32"))[:, None]
    W = torch.tensor((wt / wt.mean()).astype("float32"))[:, None]
    hub = torch.nn.HuberLoss(reduction="none", delta=12 / 30)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Y))
        for i in range(0, len(Y), 128):
            b = perm[i:i + 128]
            loss = (hub(net(X[b]), Y[b]) * W[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Zv)).numpy()[:, 0] * 30 + 50
