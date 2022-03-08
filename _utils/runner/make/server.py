def cli(configs):
    path = __file__.split('_utils')
    script_path = path[0] + '_server/sio_server.py'
    if 'server' not in configs:
        print('server not in configs')
        return None, None

    host = configs['server']['host'] if 'host' in configs['server'] else None
    port = str(configs['server']['port']) if 'port' in configs['server'] else None

    args = []
    if host:
        args.append('--host=' + host)
    if port:
        args.append('--port=' + port)
    return (script_path, args)
