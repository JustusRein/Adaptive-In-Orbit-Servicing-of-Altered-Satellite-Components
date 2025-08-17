from DToolslib import *
import os

Log = JFLogger('lalala', os.path.dirname(__file__))
Log.set_file_count_limit(1)
# Log.set_enable_trackback_exception(True)
@Inner_Decorators.time_counter
@Inner_Decorators.who_called_me
def main():
    Log.info("Hello, world!")
    try:
        1/0
    except Exception as e:
        Log.exception('ghjkghjgjh')
        
def a():
    main()

def b():
    a()
def c():
    b()

if __name__ == "__main__":
    c()
