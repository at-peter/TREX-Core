import json

def cli(configs):
    script_path = '_clients/sim_controller/sio_client.py'

    if 'server' not in configs:
        return None, None

    host = configs['server']['host'] if 'host' in configs['server'] else None
    port = str(configs['server']['port']) if 'port' in configs['server'] else None

    args = []
    if host:
        args.append('--host=' + host)
    if port:
        args.append('--port=' + port)
    args.append('--config=' + json.dumps(configs))
    return (script_path, args)
