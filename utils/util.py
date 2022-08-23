import hashlib
import uuid
import time


def getUniqueId():
    uuid_str = str(uuid.uuid4())
    md5 = hashlib.md5()
    md5.update(uuid_str.encode('utf-8'))
    return md5.hexdigest()[0:12]


def get_current_datetime():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def get_current_date():
    return time.strftime("%Y-%m-%d", time.localtime(time.time()))


def get_current_time():
    return time.strftime("%H:%M:%S", time.localtime(time.time()))