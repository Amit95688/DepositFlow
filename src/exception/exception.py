import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.logger.logger import logging

def error_message(error, error_detail_module):
    """Format an error message using the exception and a module (typically `sys`).

    This function is defensive: when `error_detail_module.exc_info()` does not
    contain traceback information (e.g. when CustomException is raised manually),
    it falls back to returning a concise message or the formatted traceback.
    """
    try:
        _, _, exc_tb = error_detail_module.exc_info()
        if exc_tb is not None:
            filename = exc_tb.tb_frame.f_code.co_filename
            line_no = exc_tb.tb_lineno
            return f"Error occurred in script: {filename} at line number: {line_no} error message: {str(error)}"
        else:
            import traceback
            tb_str = traceback.format_exc()
            if tb_str and tb_str.strip() != 'NoneType: None':
                return f"Error occurred. Traceback:\n{tb_str}"
            return f"Error: {str(error)}"
    except Exception:
        return f"Error: {str(error)}"


class CustomException(Exception):
    def __init__(self, error, error_detail_module):
        # store a human friendly message while keeping Exception semantics
        super().__init__(str(error))
        self.error_message = error_message(error, error_detail_module)

    def __str__(self):
        return self.error_message
    
if __name__=="__main__":
    try:
        logging.info("Attempting division by zero")
        a=1/0
    except Exception as e:
        logging.error(f"Exception caught: {str(e)}")
        raise CustomException(e,"Division by zero error")