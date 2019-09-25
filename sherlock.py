import jinja2
import pandas as pd
import sys
import logging
from matplotlib.pylab import plt

from pandas.errors import ParserError
from pmprophet import PMProphet
import numpy as np
import io, base64


def figure_to_base64(fig):
    io_stream = io.BytesIO()
    fig.savefig(io_stream, format='png')
    io_stream.seek(0)
    return (b'data:image/png;base64, ' + base64.b64encode(io_stream.read())).decode()


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
    # two_tailed_alpha = int(alpha / 2 * 100)
    periods = list(set([float(i.split("_")[1]) for i in self.seasonality]))

    additive_ts, multiplicative_ts = self._fit_seasonality()

    all_seasonalities = [("additive", additive_ts)]
    if len(self.multiplicative_data):
        all_seasonalities.append(("multiplicative", multiplicative_ts))
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
        all_figures = {}
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


ERRORS = {
    "no_input": "No input data file was provided",
    "not_found": "I can't find the input file you provided me, try removing any space in it's name",
    "not_readable": "The input file is not readable (make sure it's a valid csv file)",
    "missing_columns": "The input data file is missing one or more mandatory columns: {}",
    "date_column_not_readable": "The date column is not readable, try using the format dd/mm/yyyy (e.g. 30/12/1990)",
    "timespan_too_short": "Provide at least 7 days of data"
}

WARNINGS = {
    'update_not_understood': 'An update was marked as "{}" and was ignored, valid updates are "textual" and "visual"',
    "timespan_too_short": 'For best results use at least one month of data',
}

INFO = {
    'additional_regressor': 'Using column {} as an additional regressor'
}

REQUIRED_COLUMNS = [
    'date',
    'update',
    'search_downloads',
    'search_impressions'
]

OPTIONAL_COLUMNS = [
    'asa_impressions',
]


def read_input_file(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(ERRORS['not_found'])
        sys.exit()
    except ParserError:
        logging.error(ERRORS['not_readable'])
        sys.exit()
    missing_columns = ", ".join(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        logging.error(ERRORS['missing_columns'].format(missing_columns))
        sys.exit()
    try:
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    except ValueError:
        logging.error(ERRORS['date_column_not_readable'])
        sys.exit()
    return df.sort_values('date')


def run_sherlock() -> None:
    template_vars = {'seasonality': {}}

    df = read_input_file(sys.argv[1])
    for unknown_update in (set(df['update'].unique()) - set(['textual', 'visual', pd.np.nan])):
        logging.warning(WARNINGS['update_not_understood'].format(unknown_update))
    time_span = (df['date'].max() - df['date'].min()).days
    if time_span < 7:
        logging.error(ERRORS['timespan_too_short'])
        sys.exit()
    if time_span < 30:
        logging.warning(WARNINGS['timespan_too_short'])
    extra_columns = list(set(df.columns) - set(REQUIRED_COLUMNS + OPTIONAL_COLUMNS + ['date', 'search_downloads']))
    _df = df.copy()
    summary = []

    if 'visual' in df['update'].unique():
        df = _df.rename(columns={'date': 'ds'})
        if 'asa_impressions' in df.columns:
            df['impressions'] = df['search_impressions'] - df['asa_impressions']
            df['conversions'] = df['search_downloads'] - df['asa']
        else:
            df['impressions'] = df['search_impressions']
            df['conversions'] = df['search_donwloads']
        df['y'] = df['conversions'] / df['impressions']

        time_regressors = []
        for _, row in df.iterrows():
            if row['update'] == 'visual':
                additional_regressor = '{} (visual)'.format(str(row['ds']).split(" ")[0])
                df[additional_regressor] = [1 if other_row['ds'] >= row['ds'] else 0 for
                                            _, other_row in df.iterrows()]
                time_regressors.append(additional_regressor)
        m = PMProphet(
            df,
            growth=False,
            n_changepoints=0,
            seasonality_prior_scale=10,
            changepoints=[],
            name='sherlock_visual_conversion'
        )
        m.skip_first = 1000

        for time_regressor in time_regressors:
            m.add_regressor(time_regressor)

        m.add_seasonality(7, 3)
        m.fit(2000)
        prediction = m.predict(forecasting_periods=0)
        fig = plot_predict(prediction, df, [row['ds'] for _, row in df.iterrows() if row['update'] == 'visual'])
        plt.title('Conversion & Visual Updates')
        template_vars['visual_model'] = figure_to_base64(fig)
        # plt.close(fig)

        for idx, time_regressor in enumerate(time_regressors):
            error = (pd.np.percentile(
                m.trace['regressors_{}'.format(m.name)][:, idx],
                97.5
            ) - pd.np.percentile(
                m.trace['regressors_{}'.format(m.name)][:, idx],
                2.5
            )) / 2
            summary.append({
                'name': time_regressor,
                'median': pd.np.round(pd.np.median(m.trace['regressors_{}'.format(m.name)][:, idx] * 100), 2),
                'error': pd.np.round(error * 100, 2),
            })

    if 'textual' in df['update'].unique():
        df = _df.rename(columns={'date': 'ds', 'search_downloads': 'y'})
        df['y'] = df['y'] - df['asa']
        time_regressors = []
        for _, row in df.iterrows():
            if row['update'] == 'textual':
                additional_regressor = '{} (text)'.format(str(row['ds']).split(" ")[0])
                df[additional_regressor] = [other_row['y'] if other_row['ds'] >= row['ds'] else 0 for
                                            _, other_row in df.iterrows()]
                time_regressors.append(additional_regressor)

        m = PMProphet(
            df,
            growth=False,
            n_changepoints=0,
            name='sherlock_textual',
            seasonality_prior_scale=10,
            regressors_prior_scale=10
        )
        m.skip_first = 1000

        for time_regressor in time_regressors:
            m.add_regressor(time_regressor)

        for extra_column in extra_columns:
            m.add_regressor(extra_column)

        m.add_seasonality(7, 5)

        if time_span > 30:
            m.add_seasonality(30, 5)

        if time_span > 365:
            m.add_seasonality(365, 5)

        m.fit(2000)
        prediction = m.predict(forecasting_periods=0)
        fig = plot_predict(prediction, df, [row['ds'] for _, row in df.iterrows() if row['update'] == 'textual'])
        plt.title('Downloads & Textual Updates')
        template_vars['textual_model'] = figure_to_base64(fig)

        for idx, time_regressor in enumerate(time_regressors + list(extra_columns)):
            error = (pd.np.percentile(
                m.trace['regressors_{}'.format(m.name)][:, idx],
                97.5
            ) - pd.np.percentile(
                m.trace['regressors_{}'.format(m.name)][:, idx],
                2.5
            )) / 2

            summary.append({
                'name': time_regressor,
                'median': pd.np.round(pd.np.median(m.trace['regressors_{}'.format(m.name)][:, idx] * 100), 2),
                'error': pd.np.round(error * 100, 2),
            })
        extra_regressors_plots = []
        for i in range(len(time_regressors), len(time_regressors) + len(extra_columns)):
            fig = plt.figure()
            plt.grid()
            plt.hist(m.trace['regressors_{}'.format(m.name)][:, idx] * 100, bins=30, alpha=0.8, histtype='stepfilled')
            plt.axvline(np.median(m.trace['regressors_{}'.format(m.name)][:, idx]) * 100, color="C3", lw=1, ls="dotted")
            plt.title("{} (in %)".format(extra_columns[i - len(time_regressors)]))
            extra_regressors_plots.append({
                'name': extra_columns[i - len(time_regressors)],
                'img_data': figure_to_base64(fig)
            })

        template_vars['extra_regressors_plots'] = extra_regressors_plots

        seasonality = {}
        for period, fig in plot_seasonality(m, alpha=0.05, plot_kwargs={}).items():
            seasonality[int(period)] = figure_to_base64(fig)
        template_vars['seasonality'] = seasonality
    template_vars['summary'] = summary

    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template("template.html")
    outputText = template.render(**template_vars)

    open('report.html', 'w').write(outputText)
    return outputText


if __name__ == '__main__':
    if len(sys.argv) < 2:
        logging.error(ERRORS['no_input'])
        sys.exit()
    if len(sys.argv) == 3 and sys.argv[2] == '--no-asa':
        OPTIONAL_COLUMNS.append('asa')
    run_sherlock()
