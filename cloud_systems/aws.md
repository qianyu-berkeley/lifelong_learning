# AWS Tool Summary

## `Amazon S3`: Object Storage for your data

* For Data lake, decouple from compute (EC2, EMR, reshift, etc)
* Max object size 5TB
* Object Tags (key/value pair) for security and lifecycle
* Create partition for speed-up query (e.g. by date, year/month/day/hour, by product)
* Encryption for objects for ML
* SSE-S3 use key handled managed by AWS
* SSE-KMS use AWS key management service to manage encrytion keys (additional security and edit trail)


## `Kinesis`: Real-time data stream for real-time application, need capacity planning

* `Kinesis` is managed alternative to Apache Kafka
* `Kinesis Streams`: streaming ingest at scale
* `kinesis analytics`: real-time analytics on streams using SQL
* `kinesis firhose`: load streams into S3, readshift, elasticsearch or splunk
* `Kinesis Video Streams`: real-time video feeds

## `Glue`

### `Glue Data Catalog`

* `Glue Data Catalog & Crawlers`: Metadata repositories for all tables and datasets
  * Automated schema inference
  * schemas are versioned
  * Integrated with Athena and Redsfhit Spectrum
* `Glue Crawler` can help build Glue data catalog
  * Work with S3, Redshift, RDS
  * Work with Json, parquet, csv, etct data

### `Glue ETL`

* Transform, clean and enrich data, jobs run a serverless Spark platform
  * DropFields
  * Filter
  * Join
  * Map
* Glue scheduler to schedule the jobs
* Glue trigger to automate event based job run

## `Redshift`

* `Redshift`: Data Warehousing for OLAP, SQL language, for structured data
* `Redshift Spectrum`: Redshift on data in S3 (without the need to load it first in Redshift) for unstructured data

## AWS `Data Pipeline` (based on AWS `SWF`)

* Orchestration service
* User has control of compute resources (compare to Glue ETL)
* Orchestration of ETL jobs between RDS, DynamoDB, S3. runs on EC2 instances

## AWS `Batch`

* `serverless`
* Run batch jobs as Docker containers (must provide docker image)
* For ETL and non-ETL work
* schedule batch jobs using CloudWatch Events
* Orchestrate Batch jobs using AWS `Step Functions`
  * `Step Functions` to design workflow

## AWS `Athena`

* Query service for S3
* Presto based
* Serverless
* support structured, unstructured, and semi-structured

## AWS `EMR` (Elastic MapReduce)

* Hadoop on EC2
* Support Hive, Spark


## `Sagemaker`: build to handle the entire ML workflow

Fetch, clean, prepare data => Train and evaluate model => Deploy model and evaluate results in production

### Data Prepare

* Copy data from S3
* Start processing container (Sagemaker build-in or user provided)
* Output processed data to S3

### Training Job

* Training job needs:
  * Training data S3 URL
  * ML compute resources
  * URL of S3 for output
  * ECR path to training code
* Training options:
  * Sagemaker Build in algorithm
  * Integrated popular python frameworks (TF, Pytorch, etc)
  * User customer Docker image
* HyperParameter Tuning job (parallel jobs)

### Deploying models

* Save model to s3
* Deploy to a persistent endpoint
* Batch prediction of full dataset


### `Sagemaker Studio`

* Notebooks
* Experiments
* Debugger
* AutoPilot (AutoML)
* Explainability
* Model Monitor (Data drift, model performance, detect outliers and anomalies)
  * Alert via `CloudWatch
  * Integrate with `Clarify to detect bais
  * Data store in S3
  * Monitoring jobs scheduled with monitoring schedule
  * metrics are emitted to cloudwatch, based on events you can trigger actions
  * It can also integrate with other dashboard (tensorboard, quicksights, tabeau)

### Sagmekaer ML operation

* Sagemaker + Docker
  * Model hosted in Docker comtainers
  * There are pre-build model containers to use
  * User can bring their own and extend an pre-build container
  * Docker images are saived in AWS ECR
  * To make container compatible iwth SageMaker, 

    add to dockerfile
  
    ```docker
    RUN pip install sagmaker-containers
    ```
    
    Set up structure of the training container as

    ```bash
    /opt/ml
    ├── input
    │ ├── config
    │ │ ├── hyperparameters.json
    │ │ └── resourceConfig.json
    │ └── data
    │       └── < channel_name>
    │           └── <input data>
    ├── model
    │
    ├── code
    │     └── <script files>
    │
    └── output
    └── failure
    ```

    set up structure of deployment container

    ```bash
    /opt/ml
    └── model
        └── <model files>
    ```

    Required workdir
    * nginx.conf
    * predictor.py
    * serve/
    * train/
    * wsgi.py

    Dockerfile example

    ```docker
    FROM
    tensorflow /tensorflow:2.4
    RUN pip install sagemaker containers

    # Copies the training code inside the container
    COPY train.py /opt/ml/code/train.py

    # Defines train.py as script entrypoint
    ENV SAGEMAKER_PROGRAM train.py
    ```

    There other Environment setup (See AWS doc and [github example](https://github.com/aws/amazon-sagemaker-examples/tree/main/advanced_functionality))

* SageMaker + Kubernets

* Sagemaker has operators for kubenets
* Sagemaker has components for kubeflow pipelines
  * Sagemaker processing
  * training
  * inference
  * hyperparameter tuning

* SageMaker Projects (SageMaker Studio native MLOps solution with CI/CD)
  * Build images
  * prep data, feature engineering
  * train/evaluate models
  * deploy models
  * monitor and update
  * use github repo and AWS codepipline for CI/CD to build/deployment ML solution
  * Use sagemaker pipelines to define steps


## Other Services
* `VPC Endpoint Gateway`: Privately access your S3 bucket without going through the public internet
* `DynamoDB`: NoSQL store
* `RDS / Aurora`: Relational Data Store for OLTP, SQL language
* `ElasticSearch`: index for your data, search capability, clickstream analytics
* `ElastiCache`: data cache technology
* `DMS`: Database Migration Service, 1-to-1 CDC replication, no ETL
* `Step Functions`: Orchestration of workflows, audit, retry mechanisms
* `Quicksight`: Visualization Tool
* `Rekognition`: ML Service
* `DeepLens`: camera by Amazon