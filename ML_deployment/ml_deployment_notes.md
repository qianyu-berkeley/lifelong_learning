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