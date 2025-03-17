## Background

This branch was used to generate the benchmarking data for this GitHub issue:
https://github.com/opensearch-project/remote-vector-index-builder/issues/23

To run the experiments yourself, follow the steps:

### Build Core image 
1. Clone the repository to local computer: `git clone https://github.com/rchitale7/remote-vector-index-builder.git`
2. Checkout the `benchmarking` branch: `git checkout benchmarking`
3. Build the `core` image:
    - `cd remote_vector_index_builder`
    - `docker build -t core .`

Note that any time you make a change to the `core` image, you need to rebuild the `core` image.

### Push Docker images to Dockerhub
Make sure you can access your docker account through here: https://hub.docker.com/
1. Create DockerHub repository in personal account
    - Make sure repository is public
2. Tag the core image: `docker tag core [your_docker_repository]:core`
    - You can use a different repository tag instead of `:core`, if you want
3. Push the core image: `docker push [your_docker_repository]:core`


### Run Docker image

1. Provision an EC2 instance with GPUs, with the `Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)`
    - You can use an instance from the `g5` family
2. Connect to the EC2 instance, in a terminal
3. In the terminal, pull the `core` image: `docker pull [your_docker_repository]:core`
4. Create a file called `my_env`
5. Add the environment variables needed by the `core/main.py` runner, in the `my_env` file
    - If you are only benchmarking vector download, set `DOWNLOAD=True`, and 
   add the `MULTIPART_CHUNKSIZES` and `THREAD_COUNTS` you want to test. Also, set the `S3_BUCKET` variable to your s3 bucket. 
    - If you are only benchmarking index upload, set `UPLOAD=True`, `LOCAL_PATH` to the path of the files you want to try uploading. 
   And then set the `MULTIPART_CHUNKSIZES`, `THREAD_COUNTS`, and `S3_BUCKET` variables
    - If you are benchmarking the full flow, set `DOWNLOAD=True`, `BUILD=True`, `UPLOAD=True`, `MULTIPART_CHUNKSIZES`, `THREAD_COUNTS`, 
   and `S3_BUCKET`. 
6. Make sure the `knn_vec` and `knn_did` files that are being passed in `main()` function in `core/main.py` are present in your s3 bucket. 
7. Run the container: `docker run --env-file my_env [your_docker_repository]:core`
    - If you are just benchmarking the `UPLOAD` functionality, mount your directory where your files to upload are:
   `docker run -v <host_dir>:<container_dir> --env-file my_env [your_docker_repository]:core`
    - You can also run the container in the background by adding the `-d` option to the docker container
8. Results will be generated in csv file format, and pushed to s3 bucket
