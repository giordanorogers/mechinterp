import inspect
import re

def printer(variable):
    # Get the calling frame
    frame = inspect.currentframe().f_back
    # Get the source line that called this function
    line = inspect.getframeinfo(frame).code_context[0].strip()
    # Extract variable name using regex
    match = re.search(r'printer\s*\(\s*([^)]+)\s*\)', line)
    if match:
        var_name = match.group(1).strip()
        print(f"{var_name}: {variable}")
    else:
        print(f"unknown: {variable}")