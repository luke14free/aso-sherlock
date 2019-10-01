import pandas as pd
from matplotlib.pylab import plt

import numpy as np
import io, base64


def figure_to_base64(fig):
    io_stream = io.BytesIO()
    fig.savefig(io_stream, format='png')
    io_stream.seek(0)
    return (b'data:image/png;base64, ' + base64.b64encode(io_stream.read())).decode()


def plot_nowcast(model, updates):
    fig = plt.figure(figsize=(20, 10))
    y = model.trace['y_hat_%s' % model.name]
    ddf = pd.DataFrame(
        [
            np.percentile(y, 50, axis=0),
            np.max(y, axis=0),
            np.min(y, axis=0),
        ]
    ).T
    ddf["ds"] = model.data["ds"]
    ddf.columns = ["y_hat", "y_low", "y_high", "ds"]
    ddf["orig_y"] = model.data["y"]
    ddf.plot("ds", "y_hat", ax=plt.gca())
    plt.fill_between(
        ddf["ds"].values,
        ddf["y_low"].values.astype(float),
        ddf["y_high"].values.astype(float),
        alpha=0.3,
    )
    ddf.plot("ds", "orig_y", style="k.", ax=plt.gca(), alpha=0.3)
    for update in updates:
        plt.axvline(
            update, color="C3", lw=1, ls="dotted"
        )
    plt.grid(axis="y")
    return fig


def plot_predict(prediction, original, updates):
    fig = plt.figure(figsize=(20, 10))
    prediction.plot("ds", "y_hat", ax=plt.gca())
    prediction["orig_y"] = original["y"]
    plt.fill_between(
        prediction["ds"].values,
        prediction["y_low"].values.astype(float),
        prediction["y_high"].values.astype(float),
        alpha=0.3,
    )

    prediction.plot("ds", "orig_y", style="k.", ax=plt.gca(), alpha=0.2)
    for update in updates:
        plt.axvline(
            update, color="C3", lw=1, ls="dotted"
        )
    plt.grid(axis="y")
    return fig


def plot_seasonality(self, alpha: float, plot_kwargs: bool):
    periods = list(set([float(i.split("_")[1]) for i in self.seasonality]))

    additive_ts, multiplicative_ts = self._fit_seasonality()

    all_seasonalities = [("additive", additive_ts)]
    if len(self.multiplicative_data):
        all_seasonalities.append(("multiplicative", multiplicative_ts))
    all_figures = {}

    for sn, ts in all_seasonalities:
        if (sn == "multiplicative" and np.sum(ts) == 1) or (
                sn == "additive" and np.sum(ts) == 0
        ):
            continue
        ddf = pd.DataFrame(
            np.vstack(
                [
                    np.percentile(ts[:, :, self.skip_first:], 50, axis=-1),
                    np.percentile(
                        ts[:, :, self.skip_first:], alpha / 2 * 100, axis=-1
                    ),
                    np.percentile(
                        ts[:, :, self.skip_first:], (1 - alpha / 2) * 100, axis=-1
                    ),
                ]
            ).T,
            columns=[
                "%s_%s" % (p, l)
                for l in ["mid", "low", "high"]
                for p in periods[::-1]
            ],
        )
        ddf.loc[:, "ds"] = self.data["ds"]

        for period in periods:
            if int(period) == 0:
                step = int(
                    self.data["ds"].diff().mean().total_seconds() // float(period)
                )
            else:
                step = int(period)
            graph = ddf.head(step)
            if period == 7:
                ddf.loc[:, "dow"] = [i for i in ddf["ds"].dt.weekday]
                graph = (
                    ddf[
                        [
                            "dow",
                            "%s_low" % period,
                            "%s_mid" % period,
                            "%s_high" % period,
                        ]
                    ]
                        .groupby("dow")
                        .mean()
                        .sort_values("dow")
                )
                graph.loc[:, "ds"] = [
                    ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i]
                    for i in graph.index
                ]
                graph = graph.sort_index()
            fig = plt.figure(**plot_kwargs)
            all_figures[period] = fig
            graph.plot(
                y="%s_mid" % period, x="ds", color="C0", legend=False, ax=plt.gca()
            )
            plt.grid()

            if period == 7:
                plt.xticks(range(7), graph["ds"].values)
                plt.fill_between(
                    np.arange(0, 7),
                    graph["%s_low" % period].values.astype(float),
                    graph["%s_high" % period].values.astype(float),
                    alpha=0.3,
                )
            else:
                plt.fill_between(
                    graph["ds"].values,
                    graph["%s_low" % period].values.astype(float),
                    graph["%s_high" % period].values.astype(float),
                    alpha=0.3,
                )

            plt.title("Model Seasonality (%s) for period: %s days" % (sn, period))
            plt.gca().xaxis.label.set_visible(False)
    return all_figures


def safe_mean(x):
    try:
        return np.mean(x)
    except TypeError:
        x = x.dropna()
        if x.empty:
            return None
        else:
            return x[0]
