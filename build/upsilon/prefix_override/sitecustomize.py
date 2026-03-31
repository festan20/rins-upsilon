import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/upsilon/colcon_ws/rins-upsilon/install/upsilon'
