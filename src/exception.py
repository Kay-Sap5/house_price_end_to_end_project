import os
import sys
from src.logger import logging



def error_message_detail(error , error_detail : sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured file_name [{file_name}] line_number [{exc_tb.tb_lineno}] message [{str(error)}]"
    logging.info(f"{error_message}")

    return error_message

class CustomExpection(Exception):
    def __init__(self , error_message , error_detail :sys):
        super().__init__()
        self.error_message_detail = error_message_detail(error=error_message , error_detail=error_detail)

    def __str__(self):
        return self.error_message_detail


        


