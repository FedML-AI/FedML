# gunicorn.conf

# workers number
workers = 1
# worker threads
threads = 2
# ip and port
bind = '0.0.0.0:5000'
# daemon process ,manage by supervisor
daemon = 'false'
# gevent
worker_class = 'gevent'
worker_connections = 2000
# process id
pidfile = './gunicorn.pid'
# set accesslog and errorlog path
accesslog = './logs/gunicorn_acess.log'
errorlog = './logs/gunicorn_error.log'
# set loglevel
loglevel = 'info'