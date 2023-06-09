# ML Deployment Notes

## Approach 1. Flask Rest API on Cloud Instance

* Create an instances (AWS EC2 or GCP VM)
* Install environment (Python, packages)
* Pull model pickle files
* Apply python flask service code template


## Approach 2. Serverless ML API

* Use serverless service (AWS Lambda or GCP cloud functions)
* Store pickle files to storage (AWS S3, GCP object store)
* Need a python function and requirement.txt
  * copy pickle files to temp
  * Read in and predict
* Deploy

## Approach 3. User colab to host

* Same python Flask function with flask_ngrok
* Upload pickle files
* run colab notebook

**Note: use postman for api testing**


## Approach 3: Tensorflow Serving

* High performance model serving capability from Tensorflow
* With Docker
  * Create a VM instance
  * Install docker
  * Pull tensorflow/serving docker image

    ```bash
    docker pull tensorflow/serving
    docker run -t --rm -p 8080:8080 -v "$(pwd)/my_model:/models/my_model" -e MODEL_NAME=my_model tensorflow/serving
    ```

  * To access TF Serving API
  
    ```python
    import requests
    import json
    import pickle

    #url = 'http://35.238.92.68:8501/v1/models/customer_behavior_model:predict'
    url = "http://public_ipaddress_of_vm_instance/v1/models/my_model:predict'

    scaler_colab = pickle.load(open('sc.pickle','rb'))

    request_data = json.dumps({"signature_name": "serving_default", "instances": [[-1.43318661, -0.47466685],[0.2345214460208433, 0.03675871227617118]]})

    json_response = requests.post(url,request_data)
    print (json_response.text)
    ```

## Approach 4. Serverless for Tensorflow Models or pytorch

* Need to save the model weights instead of model files in a directory structure to the object store
* In the function (AWS lambda or gcp cloud function):
  * download weights from object store to `/tmp/variables.index` and `/tmpe/variables.data-xxxx`
  * define a model object
  * load_wegiht("/temp/variables")
  * The rest of step is the same as other models
* If we use pytorch, we save the model in `.pt`, run `model.load_state_dict(torch.load('/tmp/model_pytorch'))


## Approach 5. Tensorflow Javascript (TensorFlow.js)

* Perform prediction on web page directly (more secure since it is not rely on server)
* Export a tensorflow model to TensorFlow.js  (We can also build a model direction with Javascript TensorFlow.js)


### Javascript Primer

* vscode app extension live server can run website locally as it is ran from a web server
* save tensorflow model use tfjs module as a json file
* we load the model in javascript with `tf.loadLayersModule()`


## Other approaches

### Deploy model as model code

* Example: Save linear regression parameters
* extract the parameter and create a model class

### Store model in database

* we can also store a model (pickel) binary to a postgres database

## MLOps

* constant training, deployment, evaluation, retraining, monitoring (experiment tracking)
* ML Lifecycle
  1. Data Sourcing (Raw Data): `Airflow`, `NiFi`
  2. Data Preprocessing: `Spark`, `Pandas`
  3. Feature Engineering: `PyOD`, `Feast`
  4. Model Training: `Scikit-Learn` , `TensorFlow`, `Pytorch`
  5. Model evaluation, experient tracking, registry: `MLFlow`
  6. Model Deployment: `TF Serviing`, `Flask Reset API`, `serverless`
  7. Model Monitoring: `MLWatcher`

### ML Flow

* MLflow Tracking
  * Experiments => Runs => (code version, start/end time, source, parameters, metrics, tags, artifacts)
* MLflow Projects
* MLflow Models
* Model Regsitries
* Model Management enables:
  * Container-based REST servers
  * Continuous deployment using Spark streaming
  * Batch
  * Managed cloud platforms such as Azure ML and AWS SageMaker
  * **Packaging the final model in a platform-agnostic way offers the most flexibility in deployment options and allows for model reuse across a number of platforms.**

* Model packaging
  * The main abstraction in this package is the concept of flavors
    * A flavor is a different ways the model can be used
    * For instance, a TensorFlow model can be loaded as a TensorFlow DAG or as a Python function
    * Using an MLflow model convention allows for both of these flavors
* The difference between projects and models is that models are for inference and serving
* The python_function flavor of models gives a generic way of bundling models
  * Building [flavors](https://mlflow.org/docs/latest/python_api/index.html):
    * mlflow.pyfunc ([ref](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-create-custom))
    * mlflow.keras
    * mlflow.pytorch
    * mlflow.sklearn
    * mlflow.spark
    * mlflow.tensorflow 
    * Their meta information is store in `MLmodel` as part of mlflow model artifact
  * `pyfunc` is a generic object can be deployed using any platform including mlflow, Sagemaker, Spark UDF, etc
    * [XGboost Ref](https://github.com/mlflow/mlflow/blob/master/docs/source/models.rst#example-saving-an-xgboost-model-in-mlflow-format)

* Model registery and deployment stages
  * The MLflow Model Registry defines several model stages: `None`, `Staging`, `Production`, and `Archived`
  * We can use mlflow client to update model version and stage
  
    ```bash
    %sh curl -i -X POST -H "X-Databricks-Org-Id: <YOUR_ORG_ID>" -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" https://<YOUR_DATABRICKS_WORKSPACE_URL>/api/2.0/preview/mlflow/transition-requests/create -d '{"comment": "Please move this model into production!", "model_version": {"version": 1, "registered_model": {"name": "power-forecasting-model"}}, "stage": "Production"}'
    ```
    
    ```python
    client.transition_model_version_stage(
      name=model_details.name, 
      version=model_details.version, 
      stage='Production',
    )
    ```

* Stream prediction
  * create spark UDF of the model with mlflow.pyfunc
  * generate prediction on stream df
  
  ```python
  import mlflow.pyfunc

  pyfunc_udf = mlflow.pyfunc.spark_udf(spark, URI + "/random-forest-model")
  predictionsDF = streamingData.withColumn("prediction", pyfunc_udf(*streamingData.columns))

  # Wait until the stream is actually ready for processing.
  untilStreamIsReady(myStreamName)
  stopAllStreams()

  # if write to a table
  predictionsDF
    .writeStream                                           # Write the stream
    .queryName(myStreamName)                               # Name the query
    .format("delta")                                       # Use the delta format
    .partitionBy("zipcode")                                # Specify a feature to partition on
    .option("checkpointLocation", checkpointLocation)      # Specify where to log metadata
    .option("path", writePath)                             # Specify the output path
    .outputMode("append")                                  # Append new records to the output path
    .start()                                               # Start the operation
  ```


### Sagemaker

It support multiple options and multiple workflows

### 1. [Bring your own algorithm container](https://github.com/aws/amazon-sagemaker-examples/tree/main/advanced_functionality/scikit_bring_your_own)

* To train and host in Sagemaker
  * serveing with HTTP requests
* Permissions:
  * sagemakerFullAccess (To enable Sagemaker notebook or instances)
  * AmazonEC2ContinerRegisteryFullAccess (To create a new repositories in Amazon ECR)
* We can choose to use a single image for both train and host or 2 images for train or host, the decision is based on requirements and convenient
* Setup docker container
  * Because you can run the same image in training or hosting, Amazon SageMaker runs your container with the argument `train` or `serve`. How your container processes this argument depends on the container:
  * if we don't define an `ENTRYPOINT` in the Dockerfile so Docker will run the command `train` at training time and `serve` at serving time. In this case, we define these as executable Python scripts, but they could be any program that we want to start in that environment. 
  * If you specify a program as an `ENTRYPOINT` in the Dockerfile, that program will be run at startup and its first argument will be `train` or `serve`. The program can then look at that argument and decide what to do. 
  * If you are building separate containers for training and hosting (or building only for one or the other), you can define a program as an `ENTRYPOINT` in the Dockerfile and ignore (or verify) the first argument passed in.
  * Running training

    ```
    /opt/ml
    |-- input => training inputs
    |   |-- config
    |   |   |-- hyperparameters.json
    |   |   `-- resourceConfig.json
    |   `-- data
    |       `-- <channel_name>
    |           `-- <input data>
    |-- model => training model outputs, also for serving
    |   `-- <model files>
    `-- output => training log outputs
        `-- failure
    ```

  * Hosting with recommended serving stack
    HTTP request => nginx (reverse proxy) => gunicom (WSGI HTTP server) => flask (worker)