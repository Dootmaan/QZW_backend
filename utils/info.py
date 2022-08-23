import json


def success(data, message='success', status=200):
    return json.dumps({'status': status, 'message': message, 'data': data})


def error(data, message='error', status=500):
    return json.dumps({'status': status, 'message': message, 'data': data})