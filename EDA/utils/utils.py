# utils/utils.py
def dbg_print(msg, debug=False):
    """
    Print a debug message if debug is True.
    """
    if debug:
        print(f"[DEBUG] {msg}")