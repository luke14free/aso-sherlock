import numpy as np
import io
import base64

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


def figure_to_base64(fig):
    io_stream = io.BytesIO()
    fig.savefig(io_stream, format='png')
    io_stream.seek(0)
    return (b'data:image/png;base64, ' + base64.b64encode(io_stream.read())).decode()


def safe_mean(x):
    try:
        return np.mean(x)
    except TypeError:
        x = x.dropna()
        if x.empty:
            return None
        else:
            return x[0]
