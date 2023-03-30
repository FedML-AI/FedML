FROM nginx:latest AS nginx

WORKDIR /
ADD devops/dockerfile/docs-server/nginx.conf /etc/nginx/conf.d/default.conf

WORKDIR /usr/share/nginx/html/
COPY ./doc/en/_build/html/ ./

EXPOSE 80


FROM nginx:latest AS product

