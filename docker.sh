# docker build -t "ubuntu:crawler" .
docker run -it -p 80:80 -v $(pwd):/app ubuntu:crawler bash
