# utils/logging_config.py

import logging

def setup_logging():
    """
    Configure logging for the application.
    """
    logging.basicConfig(
        filename='app.log',
        level=logging.ERROR,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
