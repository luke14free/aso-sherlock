from optparse import OptionParser
from typing import Dict, List, Any, Union, Tuple

import jinja2
import pandas as pd
import sys
import logging
from matplotlib.pylab import plt

from pandas.errors import ParserError
from pmprophet import PMProphet, Sampler
import numpy as np
import pymc3 as pm

from lib.helpers import figure_to_base64, safe_mean, ERRORS, REQUIRED_COLUMNS, WARNINGS, OPTIONAL_COLUMNS
from lib.plotting import plot_nowcast, plot_seasonality


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


def handle_outliers(data: pd.DataFrame) -> pd.DataFrame:
    if options.sigma:
        y = []
        raw_y = data['y'].values
        for idx, val in enumerate(raw_y):
            if val < 0 or not (raw_y.mean() - options.sigma * raw_y.std()) < val < (
                    raw_y.mean() + options.sigma * raw_y.std()):
                ts_slice = raw_y[idx - 20:idx + 20]
                y.append(np.median(ts_slice[ts_slice >= 0]))
            else:
                y.append(val)
        data['y'] = y
    else:
        if data['y'].min() < 0:
            raise Exception(ERRORS['conversion_less_than_0'])
    return data


def fit_beta_regression(model: PMProphet, data: pd.DataFrame) -> PMProphet:
    model._prepare_fit()
    with model.model:
        mean = pm.Deterministic('y_%s' % model.name, model.y)  # no scaling needed
        hp_alpha = pm.HalfCauchy('y_alpha_%s' % model.name, 2.5)
        hp_beta = pm.Deterministic('y_beta_%s' % model.name, hp_alpha * ((1 - mean) / mean))
        pm.Beta("observed_%s" % model.name, hp_alpha, hp_beta, observed=data['y'])
        pm.Deterministic('y_hat_%s' % model.name, mean)
    model.fit(10000 if options.sampler == 'metropolis' else 2000,
              method=Sampler.METROPOLIS if options.sampler == 'metropolis' else Sampler.NUTS,
              finalize=False,
              step_kwargs={'compute_convergence_checks': False} if options.sampler == 'metropolis' else {})
    return model


def summary_from_model_regressors(model: PMProphet, regressors: Union[List, Tuple]) -> List[
    Dict[str, Union[str, float]]]:
    alpha = options.alpha
    summary = []
    for idx, regressor in enumerate(regressors):
        error = (pd.np.percentile(
            model.trace['regressors_{}'.format(model.name)][:, idx],
            100 - (alpha * 100 / 2)
        ) - pd.np.percentile(
            model.trace['regressors_{}'.format(model.name)][:, idx],
            (alpha * 100 / 2)
        )) / 2
        summary.append({
            'name': regressor,
            'median': pd.np.round(pd.np.median(model.trace['regressors_{}'.format(model.name)][:, idx] * 100), 2),
            'error': pd.np.round(error * 100, 2),
        })
    return summary


def create_model(model_name: str, data: pd.DataFrame, growth: bool, regressors: Union[List, Tuple] = (),
                 changepoints: Union[List, Tuple] = ()) -> PMProphet:
    model = PMProphet(
        data,
        growth=growth,
        seasonality_prior_scale=options.seasonality_scale,
        changepoints=[] if not changepoints else changepoints,
        name=model_name,
    )
    # model.skip_first = 200 if options.sampler == 'nuts' else 10000

    for regressor in regressors:
        model.add_regressor(regressor)

    if not options.weekly:
        model.add_seasonality(7, 3)

    if (data['ds'].max() - data['ds'].min()).days > 365:
        model.add_seasonality(365, 5)
    return model


def visual_update_analysis(df: pd.DataFrame) -> Tuple[Dict[str, str], List[
    Dict[str, Union[str, float]]]]:
    summary = []
    template_vars = {}
    df = df.rename(columns={'date': 'ds'})
    if 'asa_impressions' in df.columns:
        df['impressions'] = df['search_impressions'] - df['asa_impressions']
        df['conversions'] = df['search_downloads'] - df['asa']
    else:
        df['impressions'] = df['search_impressions']
        df['conversions'] = df['search_downloads']
    df['y'] = df['conversions'] / df['impressions']
    df.index = df['ds']

    if options.weekly:
        df = df.resample('W').apply(safe_mean)

    df = handle_outliers(df.copy())

    time_regressors = []
    for _, row in df.iterrows():
        if row['update'] == 'visual':
            additional_regressor = '{} (visual)'.format(str(row['ds']).split(" ")[0])
            df[additional_regressor] = [1 if other_row['ds'] >= row['ds'] else 0 for
                                        _, other_row in df.iterrows()]
            time_regressors.append(additional_regressor)

    model = create_model('sherlock_visual', df, False, time_regressors)
    conversion_model = fit_beta_regression(model, df)

    fig = plot_nowcast(conversion_model, [row['ds'] for _, row in df.iterrows() if row['update'] == 'visual'])
    plt.title('Conversion & Visual Updates')
    template_vars['visual_model'] = figure_to_base64(fig)

    summary.extend(summary_from_model_regressors(conversion_model, time_regressors))
    seasonality = {}
    for period, fig in plot_seasonality(conversion_model, alpha=options.alpha, plot_kwargs={}).items():
        seasonality[int(period)] = figure_to_base64(fig)
    template_vars['conversion_seasonality'] = seasonality

    return template_vars, summary


def textual_update_analysis(df: pd.DataFrame, extra_columns: List) -> Tuple[Dict[str, str], List[
    Dict[str, Union[str, float]]]]:
    template_vars: Dict[str, Any] = {}
    summary = []
    df = df.rename(columns={'date': 'ds', 'search_downloads': 'y'})
    if 'asa' in df.columns:
        df['y'] = df['y'] - df['asa']
    df.index = df['ds']

    df = handle_outliers(df)

    if options.weekly:
        df = df.resample('W').apply(safe_mean)

    time_regressors = []
    for _, row in df.iterrows():
        if row['update'] == 'textual':
            additional_regressor = '{} (text)'.format(str(row['ds']).split(" ")[0])
            df[additional_regressor] = [other_row['y'] if other_row['ds'] >= row['ds'] else 0 for
                                        _, other_row in df.iterrows()]
            time_regressors.append(additional_regressor)

    model = create_model('sherlock_textual', df, True, time_regressors + extra_columns)

    model.fit(10000 if options.sampler == 'metropolis' else 2000,
              method=Sampler.METROPOLIS if options.sampler == 'metropolis' else Sampler.NUTS,
              step_kwargs={'compute_convergence_checks': False} if options.sampler == 'metropolis' else {})

    fig = plot_nowcast(model, [row['ds'] for _, row in df.iterrows() if row['update'] == 'textual'])
    plt.title('Downloads & Textual Updates')
    template_vars['textual_model'] = figure_to_base64(fig)

    summary.extend(summary_from_model_regressors(model, time_regressors + extra_columns))

    extra_regressors_plots: List[Dict[str, str]] = []
    for i in range(len(time_regressors), len(time_regressors) + len(extra_columns)):
        fig = plt.figure()
        plt.grid()
        plt.hist(model.trace['regressors_{}'.format(model.name)][:, i] * 100, bins=30, alpha=0.8, histtype='stepfilled',
                 density=True)
        plt.axvline(np.median(model.trace['regressors_{}'.format(model.name)][:, i]) * 100, color="C3", lw=1,
                    ls="dotted")
        plt.title("{} (in %)".format(extra_columns[i - len(time_regressors)]))
        extra_regressors_plots.append({
            'name': extra_columns[i - len(time_regressors)],
            'img_data': figure_to_base64(fig)
        })

    template_vars['extra_regressors_plots'] = extra_regressors_plots

    seasonality = {}
    for period, fig in plot_seasonality(model, alpha=options.alpha, plot_kwargs={}).items():
        seasonality[int(period)] = figure_to_base64(fig)
    template_vars['textual_seasonality'] = seasonality

    return template_vars, summary


def run_sherlock() -> None:
    template_vars = {'textual_seasonality': {}, 'conversion_seasonality': {}}

    df = read_input_file(options.input_file)
    for unknown_update in (set(df['update'].unique()) - {'textual', 'visual', pd.np.nan}):
        logging.warning(WARNINGS['update_not_understood'].format(unknown_update))
    time_span = (df['date'].max() - df['date'].min()).days
    if time_span < 7:
        logging.error(ERRORS['timespan_too_short'])
        sys.exit()
    if time_span < 30:
        logging.warning(WARNINGS['timespan_too_short'])
    extra_columns = list(set(df.columns) - set(REQUIRED_COLUMNS + OPTIONAL_COLUMNS + ['date', 'search_downloads']))
    summary = []

    tv, s = visual_update_analysis(df.copy())
    template_vars.update(tv)
    summary.extend(s)

    tv, s = textual_update_analysis(df.copy(), extra_columns)
    template_vars.update(tv)
    summary.extend(s)

    if options.app_name:
        template_vars['app_name'] = options.app_name
    template_vars['summary'] = summary

    template_env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath="./lib"))
    template = template_env.get_template("template.html")

    with open(options.output_file, 'w') as output_file:
        output_file.write(template.render(**template_vars))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-a", "--app-name", dest='app_name', default=None,
                      help="Specify the app name if you want a more personalized report")
    parser.add_option("-i", "--input-file", dest="input_file",
                      help="Input CSV file", metavar="FILE")
    parser.add_option("-o", "--output-file", dest="output_file",
                      help="Output report file (in html format)", metavar="FILE", default='report.html')
    parser.add_option("-s", "--sampler", dest='sampler', choices=['metropolis', 'nuts'], default='metropolis',
                      help='Sampler to use ("nuts" is slower but more precise, default "metropolis")')
    parser.add_option("-n", "--no-asa", dest='no_asa', action="store_true", default=False,
                      help="Do not use ASA as an additional regressor (better seasonality fits)")
    parser.add_option("-w", "--weekly", dest='weekly', action="store_true", default=False,
                      help="Run the analysis on a weekly resampling")
    parser.add_option("-r", "--remove-outliers-sigma", dest='sigma', default=False, type='float',
                      help='''Remove outliers at more than X sigma from the mean (suggested values range between 1.5-3.5).
Default value is: 0 that means that Sherlock will not remove outliers''')
    parser.add_option("-l", "--significance-level", dest='alpha', default=0.05, type='float',
                      help="The significance level for the analysis (default is 0.05)")
    parser.add_option("-k", "--seasonality-scale", dest='seasonality_scale', default=5.0, type='float',
                      help="""The scale of the seasonality, if it fits poorly because you have 
great variance due to seasonality increase this""")

    (options, args) = parser.parse_args()

    if not options.input_file:
        logging.error(ERRORS['no_input'])
        sys.exit()
    if options.no_asa:
        OPTIONAL_COLUMNS.append('asa')
    run_sherlock()
