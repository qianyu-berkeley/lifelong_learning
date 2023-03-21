# Spark Programming

## Big Data Systems

* Hadoop: running a distributed cluster as a single machine needs 3 key pieces
  * YARN: cluster operatiing system (orchastrate AM containers at work nodes for different apps, manage nodes, resources, app and master)
  * HDFS: Distributed storage (split files into blocks 128MB, distribute to different nodes, maintain file metadata: filename, size, blocks etc)
  * Map/Reduce: Distributed Computing programming model and framework
    * Map function run at each node, reduce function aggregate all nodes
* Hive is a popular application of Hadoop in the past
* Apache Spark + Databricks inherit hadoop model and improved every aspects of hadoop and work in cloud env
  * Data Lake (old)
  * Lakehouse - On Cloud (new)
  * Spark does not come with cluster management system, it uses YARN, Kubernetes, Mesos as the cluster manager system. 
  * Spark also does not come with a storage management system, it uses HDFS, S3, Azure Blob, GCS, CFS etc.
  * Spark Compute Engine interact with the cluster and storage management system to manage data processing jobs

![image](reference_materials/datalake.png)

## Spark Execution Model and Architecture

* Execution model
  * Client: Interactive client (spark shell, notebook)
    * Client Machine for interactive mode (not for long running jobs)
  * Cluster: Submit job (spark-submit, databrick notebook, rest api)
* Cluster Manager 
  * **local[n]**: n = 1, driver only, n = 3, 1 driver and 2 executor (defined in `spark.conf`)
  * **YARN** (On-Premise, On-Cloud)
  * Kubernetes
  * Mesos
  * Standalone
  * Distributed processing model
    * Driver 1 => Excecutors, Driver 2 => Executors
    
## Using Spark

* local setup:
  * Need to download and `export $SPARK_HOME=downloaded_sparkpath`
  * Need to use java8 or 11 so use `export JAVA_HOME=/usr/local/Cellar/openjdk@11/11.0.16`
* GCP Dataproc cluster is a Yarn cluster (AWS EMR)
* `spark history server` show executions in the past
* `spark context UI` show event timeline
* `spark-submit --master yarn --deploy-mode cluster myfile.py 100` one can run spark-summit in spark shell

### Spark logging

* Create a log4j configuration file: create a `log4j.properties` to define how we will log information for spark apps, it allows us to collect logs from distributed executors and append together
* Configure spark JVM to pickup the log4J configuration file: Add to `spark-default.conf` with `spark.driver.extraJavaOptions` with `-Dlog4j.configuration=file:log4j.properties -Dspark.yarn.app.container.log.dir=app-logs -Dlogfile.name=myproject`
* Create a Python Class to get Spark's Log4J instance and use it

### Spark Session
* Spark session is the driver for an spark app. When we start a spark shell or databrick notebook, it will create a spark session behind the scene call `spark`


## Spark Data Frame

Read => Process => Write

### Read files

* Dataframe must have column name and their datatype

  ```python
  df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(filename)
  ```

### Partition and Executors

* Dataframe store in partition of HDFS
  * Driver create logical In-Memory partition and orchastrate cluster manager (e.g. Yarn) to assign executors
  * Executors is assigned with partitions to work on
* Spark dataframe is immutable
  * driver perform transformations, user intermediate variable to store outcome of transformation

  ```python
    filtered_df = df \
      .where("age < 40") \
      .select("age", "gender", "country", "state")
    grouped_df = filtered_df.groupBy("Country")
    count_df = grouped_df.count()
    ```

* spark data operations is a dag of operations
  * transformations (repartition)
    * narrow dependency: can perform on each partition (e.g. where)
    * wide dependency: require data from other partition (e.g. groupBy, join, orderBy, distinct) => shuffle/sort exchange
  * actions (e.g. read, write, collect, show)
  * Lazy evaluation: when we create a dag of spark data operations, driver will optimize the dag, create an execution plan (may not be the same as our code sequence)
    * Transformation is lazy
    * Action is immediate
      * `show` is to print dataframe and mainly used for debugging
      * `Collect` action return the data frame as python list

## Spark Structured Data Processing API

* RDD APIs (raw, more flexible, but not optmized by Catalyst optimizer, not recommended to use)
  * Catalyst Optimizer
  * Dataset API (strongly tie to JVM thus with Scalar and Java)
  * DataFrame API (python driven)
  * Spark SQL (SQL driven)

* Spark RDD (Resilient Distributed Dataset) API
  * Fault tolarant (contain meta data on how to recreate partition)
  * Create RDD (need to use spark context)

    ```python
    SurveyRecord = namedtuple("SurveyRecord", ["Age", "Gender", "Country", "State"])
    sc = SparkContext(conf=conf)
    # or
    sc = spark.sparkContext
    linesRDD = sc.textFile(sys.argv[1])
    partitionedRDD = linesRDD.repartition(2)
    colsRDD = partitionedRDD.map(lambda line: line.replace('"', '').split(","))
    selectRDD = colsRDD.map(lambda cols: SurveyRecords(init(cols[1]), cols[2], cols[3], cols[4]))
    kv = filteredRDD.map(lambda r: (r.Country, 1))
    countRDD = kvRDD.reduceByKey(lambda v1, v2: v1 + v2)
    countRDD.collect()
    ```
  * Raw, more flexible, but not practical

* SparkSQL API
  * Register the dataframe to a view before perform spark SQL

  ```python
  surveyDF.createOrReplaceTempView("survey_tbl)
  countDF = spark.sql("select Country, count(1) as count from survey_tbl where Age<40 group by Country")
  countDF.show()
  ```
  
  * same performance as dataframe API

* Catalyst Optimizer (Spark SQL Engine)
  1. Analysis
  2. Logical optimization
  3. Physical planning
  4. Code generation
  

## Spark Data Sources and Sinks

* Data Source (Reading)
  * External (external to datalake)
    * JDBC data source (Oracle, SQL Server PostgresSQL)
    * NoSQL data source (Cassandra, MongoDB)
    * Cloud Data Warehouses (Snowflake, Redshift)
    * Stream Integrators (kafka, Kinesis)
    * 2 approaches:
      1. bring to lake first, then read => recommneded for batch data
      2. Use spark data source api directly connect to external source => recommended for streaming data

  * Internal
    * HDFS
    * AWS S3
    * Azure Blob
    * GCP
    * File format
      * CSV
      * JSON
      * Parquet
      * AVRO
      * Plain TEXT
      * Spark SQL Table
      * Delta Lake

* Data Sink (Write)
  * External (same as above)
  * Internal (same as above)

* Spark DataFrame Reader API

  ```python
  spark.read
    .format("csv")
    .option("header", "true")
    .option("path", "path/data/")
    .option("mode", "failfast")
    .schema(mySchema)
    .load()
  ```

  * Read modes (`option("mode", "failfast")`)
    * permissive
    * dropmalformed
    * failfast



# Learning Reference:

* [Spark Programming](https://github.com/LearningJournal/Spark-Programming-In-Python.git)