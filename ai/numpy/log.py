
log_debug = 7
log_info = 6
log_level = log_info

def log_d(*info):
    if log_level == log_debug:
        print(*info)

def log_i(*info):
    if log_level == log_info:
        print(*info)

