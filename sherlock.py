from optparse import OptionParser

import jinja2
import pandas as pd
import sys
import logging
from matplotlib.pylab import plt

from pandas.errors import ParserError
from pmprophet import PMProphet, Sampler
import numpy as np
import pymc3 as pm

from helpers import plot_nowcast, figure_to_base64, safe_mean, plot_seasonality

ERRORS = {
    "no_input": "No input data file was provided",
    "not_found": "I can't find the input file you provided me, try removing any space in it's name",
    "not_readable": "The input file is not readable (make sure it's a valid csv file)",
    "missing_columns": "The input data file is missing one or more mandatory columns: {}",
    "date_column_not_readable": "The date column is not readable, try using the format dd/mm/yyyy (e.g. 30/12/1990)",
    "timespan_too_short": "Provide at least 7 days of data",
    "dls_less_than_0": "Downloads are below zero for certain dates. Check your data or allow outlier removal with -r",
    "conversion_less_than_0": "The conversion is below zero for certain dates. Check your data or allow outlier removal with -r",
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
    template_vars = {'textual_seasonality': {}, 'conversion_seasonality': {}}

    df = read_input_file(options.input_file)
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
            df['conversions'] = df['search_downloads']
        df['y'] = df['conversions'] / df['impressions']
        df.index = df['ds']

        if options.weekly:
            df = df.resample('W').apply(safe_mean)

        if options.sigma:
            y = []
            raw_y = df['y'].values
            for idx, val in enumerate(raw_y):
                if val < 0 or not (raw_y.mean() - options.sigma * raw_y.std()) < val < (
                        raw_y.mean() + options.sigma * raw_y.std()):
                    ts_slice = raw_y[idx - 20:idx + 20]
                    y.append(np.median(ts_slice[ts_slice >= 0]))
                else:
                    y.append(val)
            df['y'] = y
        else:
            if df['y'].min() < 0:
                raise Exception(ERRORS['conversion_less_than_0'])

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
            changepoints=[],
            name='sherlock_visual_conversion'
        )
        m.skip_first = 1000

        for time_regressor in time_regressors:
            m.add_regressor(time_regressor)

        if not options.weekly:
            m.add_seasonality(7, 7)

        if time_span > 365:
            m.add_seasonality(365, 5)

        m._prepare_fit()
        with m.model:
            mean = pm.Deterministic('y_%s' % m.name, m.y)  # no scaling needed
            hp_alpha = pm.HalfCauchy('y_alpha_%s' % m.name, 2.5)
            hp_beta = pm.Deterministic('y_beta_%s' % m.name, hp_alpha * ((1 - mean) / mean))
            pm.Beta("observed_%s" % m.name, hp_alpha, hp_beta, observed=df['y'])
            pm.Deterministic('y_hat_%s' % m.name, mean)

        m.fit(10000 if options.sampler == Sampler.METROPOLIS else 2000,
              method=Sampler.METROPOLIS if options.sampler == 'metropolis' else Sampler.NUTS,
              finalize=False)
        fig = plot_nowcast(m, [row['ds'] for _, row in df.iterrows() if row['update'] == 'visual'])
        plt.title('Conversion & Visual Updates')
        template_vars['visual_model'] = figure_to_base64(fig)

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

            seasonality = {}
            for period, fig in plot_seasonality(m, alpha=0.05, plot_kwargs={}).items():
                seasonality[int(period)] = figure_to_base64(fig)
            template_vars['conversion_seasonality'] = seasonality

    if 'textual' in df['update'].unique():
        df = _df.rename(columns={'date': 'ds', 'search_downloads': 'y'})
        if 'asa' in df.columns:
            df['y'] = df['y'] - df['asa']

        if options.sigma:
            y = []
            raw_y = df['y'].values
            for idx, val in enumerate(raw_y):
                if val < 0 or not (raw_y.mean() - options.sigma * raw_y.std()) < val < (
                        raw_y.mean() + options.sigma * raw_y.std()):
                    ts_slice = raw_y[idx - 20:idx + 20]
                    y.append(np.median(ts_slice[ts_slice >= 0]))
                else:
                    y.append(val)
            df['y'] = y
        else:
            if df['y'].min() < 0:
                raise Exception(ERRORS['dls_less_than_0'])

        df.index = df['ds']
        if options.weekly:
            df = df.resample('W').apply(safe_mean)

        time_regressors = []
        for _, row in df.iterrows():
            if row['update'] == 'textual':
                additional_regressor = '{} (text)'.format(str(row['ds']).split(" ")[0])
                df[additional_regressor] = [other_row['y'] if other_row['ds'] >= row['ds'] else 0 for
                                            _, other_row in df.iterrows()]
                time_regressors.append(additional_regressor)

        m = PMProphet(
            df,
            growth=True,
            name='sherlock_textual',
            positive_regressors_coefficients=True
        )
        m.skip_first = 1000

        for time_regressor in time_regressors:
            m.add_regressor(time_regressor)

        for extra_column in extra_columns:
            m.add_regressor(extra_column)

        if not options.weekly:
            m.add_seasonality(7, 7)

        if time_span > 365:
            m.add_seasonality(365, 7)

        m.fit(10000 if options.sampler == Sampler.METROPOLIS else 2000,
              method=Sampler.METROPOLIS if options.sampler == 'metropolis' else Sampler.NUTS)
        fig = plot_nowcast(m, [row['ds'] for _, row in df.iterrows() if row['update'] == 'textual'])
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
        template_vars['textual_seasonality'] = seasonality

    template_vars['summary'] = summary

    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template("template.html")
    outputText = template.render(**template_vars)

    open(options.output_file, 'w').write(outputText)
    return outputText


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--input-file", dest="input_file",
                      help="Input CSV file", metavar="FILE")
    parser.add_option("-o", "--output-file", dest="output_file",
                      help="Output report file (in html format)", metavar="FILE", default='report.html')
    parser.add_option("-s", "--sampler", dest='sampler', choices=['metropolis', 'nuts'], default='nuts',
                      help='Sampler to use ("nuts" is slower but more precise and suggested, otherwise "metropolis")')
    parser.add_option("-n", "--no-asa", dest='no_asa', action="store_true", default=False,
                      help="Do not use ASA as an additional regressor (better seasonality fits)")
    parser.add_option("-w", "--weekly", dest='weekly', action="store_true", default=False,
                      help="Run the analysis on a weekly resampling")
    parser.add_option("-r", "--remove-outliers-sigma", dest='sigma', default=False, type='float',
                      help='''Remove outliers at more than X sigma from the mean (suggested values range between 1.5-3.5).
Default value is: 0 that means that Sherlock will not remove outliers''')

    (options, args) = parser.parse_args()
    if not options.input_file:
        logging.error(ERRORS['no_input'])
        sys.exit()
    if options.no_asa:
        OPTIONAL_COLUMNS.append('asa')
    run_sherlock()
