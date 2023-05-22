# Docker

## Concepts

* Containerize applications
* Allow running each service with its own depedencies in seperate containers
* Sit on top of OS and hardware
* Container:
  * Process | network | mounts
  * Sharing kernel
  * Virtual machines is more costly, it also contains OS and sits on top of hypervisor. A virtual machine can have multiple containers
  * Life cycle: create => start => stop => start / restart => stop ...
* Image:
  * Image is package template plan. We can use an image to create mutliple containers

## Create custom docker images

### Steps

* Create a docker file
  * Flow: 
    1. Specify a base image with `FROM` (e.g. install an OS (possibly pulls from public image) with some default set of programs)
    2. Run some commands to install some other specific programs by using a temperory container, took a snapshot of temp container to a new temp image and turn off the temp container
    3. specify a command to run on container start-up by using a temerory container, took a snapshot of temp comtainer to a new final image and turn off temp container
* run `docker build ...` command
* push image to docker registry
* Docker build operation leverage cache (based on previous steps) so build next time will be much faster

## Docker file

* Dockerfile following the format of `instruction argument`
  * Most common instruction are: `FROM`, `RUN`, `CMD`
  * Example:

    ```docker
    FROM Ubuntu

    # If need a particular version of the image
    # FROM Ubuntu:6.0

    RUN apt-get update
    RUN apt-get install python

    RUN pip install flask
    RUN pip install flask-mysql

    COPY . /opt/source-code

    ENTRYPOINT FLASK_APP=/opt/source-code/app.py flask run
    ```

* Build an image from the current folder

    ```sh
    docker build .
    ```

* Build using a tag `-t tag_name` and push. The convension for tag name is `user_name/project_name:version`  

    ```sh
    docker build Dockerfile -t qianyu88/my_app:latest my_app_directory
    docker push qianyu88/my_app:latest
    ```

* When define the Docker file, we need to consider dependencies to minimize the rebuild image time. We copy only the necessary file **before installation command** and copy the files that does not impact installation after to prevent unnecessary image rebuild

* **`WORKDIR`** : The `WORKDIR` instruction sets the working directory for any `RUN`, `CMD`, `ENTRYPOINT`, `COPY` and `ADD` instructions that follow it in the Dockerfile. If the WORKDIR doesn’t exist, it will be created even if it’s not used in any subsequent Dockerfile instruction.

# Docker run (launch docker applications) details

* Run a docker container from an image
  ``` bash
  docker run qianyu88/myapp
  ```

* Run a docker image with port mapping (`-p <external_port>:<container_port>`)
  ``` bash
  docker run -p 8080:6000 qianyu88/myapp
  ```

* Run a docker container with shell enabled 
  ```bash
  docker run -it qianyu88/myapp sh

  # if the docker container is already running
  docker exec -it container_id sh        
  ```

## CMD and ENTRIPOINT

CMD format: `CMD ["command", "param1"]`
* Docker file example
    ```docker
    FROM ubuntu
    CMD ["sleep" "5"]
    ```
    with
    ```sh
    $ docker build dockerfile -t ubuntusleeper
    $ docker run ubuntusleeper
    ```
    is the same as
    ```sh
    $ docker run ubuntu sleep 5
    ```

* if we want command be more flexible at startup we can define `ENTRYPOINT`

    Docker file
    ```docker
    FROM ubuntu
    ENTRYPOINT["sleep"]
    # default value is 5
    CMD ["5"]
    ```
    to run
    ```sh
    docker run ubutusleeper 10
    ```

## Volume mapping

volume mapping allow us to map a local directory to container directory so that we don't need to rebuild the image everytime there is a new change.

```bash
# map current local folder $(pwd) to app folder in the container except the node_modules 
# for which we will use the one from the container
docker run -it -p 3000:3000 -v /app/node_modules -v $(pwd):/app <image_id>
```

## Attach to a container 

Allow us to access the stdin, stdout, stderr of a container from the terminal

```bash
docker run -it simple-prompt-docker bash 
```

# `Docker` Commands Summary

```sh
# Start a container (create and start)
docker run nginx

# display system-wide info
docker info

# List containers
docker ps
docker ps -a

# Start a container but cannot replace default command when it is first created
docker start container_id
docker start -a container_id

# Stop a container or kill a container
docker stop container_name      # send a sigterm to the container, perform clean-up then stop
docker kill container_name      # send a sigkill to the container immediately

# Rmove a container
docker rm container_name

# List images
docker images

# Remove images
docker rmi nginx

# Download an image
docker pull nginx

# Append command
docker run ubuntu sleep 5

# other container command
docker top CONTAINER                 # display runing processes of a container
docker logs CONTAINER                # fetch logs of a container without start/restart container
docker start CONTAINER               # start one or more stopped containers
docker restart CONTAINER             # start one or more stopped containers
docker exec -it CONTAINER CMD        # run command in a running container with terminal access 
docker exec CONTAINER CMD            # run command in a running container
docker inspect CONTAINER|IMAGE|TASK  # return low-level info on container, image or task
docker port CONTAINER [PRIVATE_PORT] # list port mappings for container
docker run -d container              # detach a container
docker attach container              # attach a container
docker system prune                  # rm and clean up all containers

# More run command
docker run -i simple-prompt-docker   # interactive connection by grabbing stdin of the container
docker run -it simple-prompt-docker   # interactive connection by grabbing stdin of the container and assigns a pseudo-tty or terminal inside the new container
docker run -it container sh           # run a new container and run the shell
docker run redis:4.0                  # with tag

# Port mapping - maps required network ports inside the container to host
docker run -p 80:5000 simple-webapp
docker run -p 3306:3306 mysql

# volume mapping
docker run -v /opt/datadir:/var/lib/myseql mysql

# Define environment variable for program in the container to use
docker run -e APP_COLOR=blue simple-webapp-color
```

## Docker Compose

Docker compose can be used to reduce the usage of docker cli and automate actions

We can use it to define and run multi-container apps by composing a file to configure your app's services. Using a single command, create and start all the services from your configuration. With docker-compose, docker will automatically put container (services) under the same network and they can talk to each other freely

## `docker-compose` commands

* `-f` : specify an alternateive compose file (default: `docker-compose.yml`)
* `-d` : running in the background

```sh
logs [SERVICE...]    # view output from containers
ps   [SERVICE...]    # list containers (-q displays on IDs)
rm   [SERVICE...]    # remove stopped containers
stop [SERVICE...]    # stop services
up   [SERVICE...]    # create and start containers
down [SERVICE...]    # stop containers (need to in the same folder where docker-compose.yaml file is)
```

### Example: `up`

Run containers in background (detached mode) and specify alternate `docker-compose.yml` file
```sh
$ docker-compose -f alt-docker-compose.yml up -d
```

## docker compose yaml file latest format

```yml
version: 3
serrices:
    db:
      image: postgres:9.4
      networks:
        - back-end
    redis:
      image: redis
      networks:
        - back-end
    app:
      image: myapp
      links:
         - redis
      networks:
        - back-end
        - front-end
      ports:
         - 5000:80
networks:
  front-end
  back-end
```

## docker compose yaml example 2 service web and test

### Example 1

* context define the path to the Dockerfile, `.` means current folder
* volumes says don't touch `/app/node_modules` but auto sync between `.` to `/app`
```yml
version: '3'
services:
  web:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - /app/node_modules
      - .:/app
  tests:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - /app/node_modules
      - .:/app
    command: ["npm", "run", "test"]
```

### Example 2

* To set `environment` by using `variableName=value`, we setup variable at the run time. If we just provide `variableName`, its value will take from your computer
* redis port can be get from docker redis image document
```yml
version: '3'
services:
  postgres:
    image: 'postgres:latest'
    environment:
      - POSTGRES_PASSWORD=postgres_password
  redis:
    image: 'redis:latest'
  nginx:
    depends_on:
      - api
      - client
    restart: always
    build:
      dockerfile: Dockerfile.dev
      context: ./nginx
    ports:
      - '3050:80'
  api:
    build:
      dockerfile: Dockerfile.dev
      context: ./server
    volumes:
      - /app/node_modules
      - ./server:/app
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - PGUSER=postgres
      - PGHOST=postgres
      - PGDATABASE=postgres
      - PGPASSWORD=postgres_password
      - PGPORT=5432
  client:
    stdin_open: true
    build:
      dockerfile: Dockerfile.dev
      context: ./client
    volumes:
      - /app/node_modules
      - ./client:/app
  worker:
    build:
      dockerfile: Dockerfile.dev
      context: ./worker
    volumes:
      - /app/node_modules
      - ./worker:/app
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
```


## A Simple CI/CD Flow Concept

1. github links to a CI/CD tool (e.g. travis, Jenkines)
2. Build the flow use yaml file of CI/CD tool to enable steps of operations 
3. The steps of operations is triggered by the github update (master branch by default)

### Key files and Depdencies: 

* CI/CD Yaml or Jenkinsfile => define CI/CD steps
* Dockerfiles the used the CI/CD Yaml
* Alternatively dependency on the CI/CD tool requestment, we can use docker-compose.yml to set up containers with different command in the CI/CD Yaml file
  


### `bash` completion

to enable bash completion (if using `brew`):
```sh
cd /usr/local/etc/bash_completion.d
ln -s /Applications/Docker.app/Contents/Resources/etc/docker.bash-completion
ln -s /Applications/Docker.app/Contents/Resources/etc/docker-machine.bash-completion
ln -s /Applications/Docker.app/Contents/Resources/etc/docker-compose.bash-completion
```

## CI/CD flow Docker Multi-container application 

1. push code to github
2. CICD tool (Travis, Jenkins) is triggered to pull from github latest repo
3. CICD tool build test images and run test codes
4. CICD tool build production images
5. CICD tool push images to artifactory service or docker image repository
6. CICD tool push the project to the service hosts (compute resource for the project)
7. Computer service pull images from the artifactory or docker image repo and deploys the application


## Side Notes

`--` in bash means the end of the command option to avoid flag conflict, in example below, `--coverage` is for `npm run test` not `docker run`, so we use `--` to break the docker command

```bash
docker run -e CI=true stephengrider/docker-react npm run test -- --coverage
```
