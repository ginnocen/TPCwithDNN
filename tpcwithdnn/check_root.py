"""
Check whether ROOT deps are present
"""

import sys
import importlib

def check_import(mod_name, add_message=""):
    try:
        importlib.import_module(mod_name)
        return True
    except ImportError:
        print(f"Cannot import {mod_name}.")
        if add_message:
            print(add_message)
        return False


CHECK_MODULES = ("ROOT",)
MESSAGES = ("Please source manually if not on aliceml",)

IMPORT_OK=True

for chm, m in zip(CHECK_MODULES, MESSAGES):
    IMPORT_OK = check_import(chm, m) and IMPORT_OK

if not IMPORT_OK:
    sys.exit(1)
