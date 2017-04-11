FROM ubuntu:16.04

RUN apt-get update
RUN apt-get -y install build-essential make libgoogle-perftools-dev google-perftools gcc-4.9 g++-4.9
RUN ln -sf /usr/bin/gcc-4.9 /usr/bin/gcc
RUN ln -sf /usr/bin/g++-4.9 /usr/bin/g++

WORKDIR /app
 
# CMD ["/my_app/start.sh"]
