import sys

def error_message_detail(error, error_detail: sys):
    """
    This function returns the error message with details.
    """
    _, _, exc_tb = error_detail.exc_info()
    # Extracting file name and line number from the traceback
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in Python script: [{file_name}] at line number: [{line_number}] with error message: [{str(error)}]"

    return error_message

class CustomException(Exception):
    """
    Custom exception class that inherits from the built-in Exception class.
    It is used to raise exceptions with detailed error messages.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message