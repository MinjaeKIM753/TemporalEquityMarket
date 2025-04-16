# utils/utils.py
from utils.settings import setting

def dbg_print(msg):
    if setting.debug:
        print(f"[DEBUG] {msg}")