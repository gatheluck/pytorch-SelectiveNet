from datetime import datetime

def get_time_stamp(mode='long'):
    if mode == 'long':
        time_stamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    elif mode == 'short':
        time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        raise ValueError('mode is invalid value')

    return time_stamp

if __name__ == '__main__':
    print(get_time_stamp())
    print(get_time_stamp('short'))