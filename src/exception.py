import sys 

def error_message(error,message:str):
    _,_,exc_tb=error_details= sys.exc_info()     
    filename= exc_tb.tb_frame.f_code.co_filename
    error_message= f"Error occurred in script: {filename} at line number: {exc_tb.tb_lineno} error message: {str(message)}"
    return error_message


class CustomException(Exception):
    def __init__(self,error,message):
        super().__init__(message)
        self.error_message= error_message(error,message)

    def __str__(self):
        return self.error_message