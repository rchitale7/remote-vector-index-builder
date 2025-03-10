## Run Remote Vector Index Builder Service

The Remote Vector Index Builder Service is separated into 
two Docker images. The `core` image contains the low level code
needed to download vectors, construct an index on GPUs, and 
upload the index to the Remote Store The `api` image provides out of the
box APIs that directly integrate with the OpenSearch Vector Engine, 
using the `core` image code under the hood. 

Here are the instructions for running the `api` image as a standalone container:

### Build Docker images
1. Clone the repository to local computer: `git clone https://github.com/rchitale7/remote-vector-index-builder.git`
2. Checkout the `api` branch: `git checkout api`
3. Build the `core` image first:
    - `cd remote_vector_index_builder`
    - `docker build -t core .`
4. Next, build the `api` image:
    - `cd testing_api`
    - `docker build -t api .`

Note that any time you make a change to the `core` image, you need to rebuild both the `core` and `api` images. 
However, if you only make a change to the `api` image, you need to build just the `api` image. 

### Push Docker images to Dockerhub
Make sure you can access your docker account through here: https://hub.docker.com/
1. Create DockerHub repository in personal account
    - Make sure repository is public
2. Tag the core image: `docker tag core [your_docker_repository]:core`
    - You can use a different repository tag instead of `:core`, if you want
3. Push the core image: `docker push [your_docker_repository]:core`
4. Tag the api image: `docker tag api [your_docker_repository]:api`
    - You can use a different repository tag instead of `:api`, if you want
5. Push the api image: `docker push [your_docker_repository]:api`

Note that any time you make a change to the `core` image, you need to re-push both the `core` and `api` images.
However, if you only make a change to the `api` image, you need to push just the `api` image.

### Run Docker image
For this step, you have the option of using the `api` image in your personal repository. 
Or, you can use an `api` image that has already been created here: 
https://hub.docker.com/layers/rchitale7/remote-index-build-service/api/images/sha256-b2ec0b04a95b28d151a2b293e049498c5cc92d19a88837019fe05a96e045e4e6
If you don't plan on making any changes to the `api` or `core` images, you can use the pre-created image. 

1. Provision an EC2 instance with GPUs, with the `Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)`
    - You can use an instance from the `g5` family
2. Connect to the EC2 instance, in two separate terminals
3. In one terminal, pull the `api` image: `docker pull [your_docker_repository]:api`
    - For the pre-created image: `docker pull rchitale7/remote-index-build-service:api`
4. Start the `api` container: `docker run --gpus all -p 80:80 [your_docker_repository]:api` 
5. In the other terminal, issue a build request:
    ```
    curl -XPOST "http://0.0.0.0:80/_build" \
    -H 'Content-Type: application/json' \
    -d '
        {
            "repository_type": "s3",
            "container_name": "<your_s3_bucket>>",
            "vector_path": "<vector_path_in_s3_bucket>",
            "doc_id_path": "<doc_id_path_in_s3_bucket>",
            "dimension": "<vector dimension>",
            "doc_count": "<number of vectors>"
        }
    '
    ```
    This will return a job id, if the build request was successfully submitted
6. Check the status of your build request: `curl -XGET "http://0.0.0.0:80/_status/<job_id>"`