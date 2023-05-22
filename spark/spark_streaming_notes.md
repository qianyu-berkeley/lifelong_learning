# Spark Real-time Stream Processing

## Spark Structured Streaming Intro

* Batch processing is a subset of streaming processing which extends spark batch processing
* Stream processing solve batch processing problems
  * Check points
  * Tracking last processed files
  * fault tolarance

## Spark Streaming framework

* Micro-batch
  * Autolooping between micro-batches
  * batch start, end position management
  * Intermediate state management
  * combine results to the prevous batch results
  * fault tolarance and restart management

## Spark Structured Streaming API

* Dataframe based streaming API
* SQL Engine optimization
* Support event time semantic
* Future enhancement and features
* 3 Step processes
  * Read streaming source  - input dataframe
  * Data Transform - process and transform input dataframe to output dataframe
  * Write to output - streaming sink
* readStream 
  * take options of input source (e.g. `socket`, `kafka`, `json`) use option of `host`, `port`, `path`)
  * Can perform operations of standard spark dataframe
* writeStream take mandortary options of `checkpointLocation` and outputMode (e.g. `complete`)
* Stream operation typically is expect to run forever until manual stop / kill or exception
* useful spark session options:
  * use `.config("spark.streaming.stopGracefullyOnShutdown", "true")` to allow stop gracefully
  * use `.config("spark.sql.shuffle.partitions", "3")` (default is 200) since we group a very small set of data
  * use `.config("spark.sql.streaming.schemaInference", "true")` to allow readStream to infer schema

## Spark Streaming processing model

* A loop of microbatch
  * execution plan based on the code
  * create spark streaming background thread
  * Trigger a microbatch
    * Read => Transform => Sink
    * There are multiple ways to trigger
      * Unspecified (default): new micro batch will trigger as soon as the current one completes and new data is available
      * Time interval: only start trigger x mins from the start of the current microbatch, 
        * if the current batch is not complete, it will wait until it completes
        * if the current batch is complete before x mins, it will wait until x mins then looking for new file
      * One time: same as batch job
      * Continuous: to achieve milli-second latency (new)

## Streaming data source

* Support 4 sources:
  * Socket source: reading from a socket connection, not for production, only for learning (e.g ncap)
  * Rate source: Dummy data source based on defined the rate
  * File source
    * Use case: `exactly once processing`, handle failure scenario, no missing, no duplication
    * If defined file type, need to ensure input path only contain this file type to avoid exception
    * Options:
      * `maxFilesPerTrigger`: limit the number of file per microbatch
      * `cleanSource`, `sourceArchieveDir` options allow to archieve processed file automatically and move to the defined archieve location
        * It Add delay to the processing time of microbatch
        * But too much input process file (as it accumulate) would also impact the microbatch, need to periodically clean. We can use a seperate process to clean input
    * Typically minute-based micro-batch
  * Kafka source
    * By default kafka consumer read from the latest offset, change to `earliest` allows it to read from the beginning at the start
    * Spark maintain the current offset in the checkpoint, startingOffsets will be overwritten in the checkpoint
    * kafka value is in binary, we need to deserialization
      * define a schema for the data
      * cast binary to string (knowing the input data format)
        * String: cast()
        * csv: from_csv()
        * json: from_json()
        * AVRO: from_avro()
          * avro require a seperate spark dependency that need to defined in spark config
          * avro require a seperate schema definition (e.g. a seperate schema file)
      * To Run kafka:
        * Start zookeeper
        * Start kafka
        * Create a topic
        * Create producer and send data
        * run pyspark streaming
      * To print schema for testing, we can use `spark.read` instead of `readStream` for kafka data source
      * Send data to Kafka (need to serialization)
        * to_json(struct(*columns))
        * to_csv(struct(*columns))
        * to_avro(struc(*columns))

## output mode

* `Append`: **insert only**, no update on data from previous microbatch, does not work with aggregation since it does not make sense if only insert
* `Update`: **upsert** like operation
* `Compele`: **overwrite** the complete results

## Fault tolerance and restart

* Streaming operation must be able to handle stop and start operation (e.g. maintenance or failure)
* `restart with exactly once` feature
  * do not miss any input records
  * do not create duplicated records
* maintain the state of each microbatch in the checkpoint location
  * Checkpoints contains read position (i.e start and end of data range) and state information
* To safely restart, we need to:
  * restart at the same checkpoint location
  * use the replayable source
  * use the deterministic computation
  * use an idempotent sink (no change, no duplication)
* Bug fixes and restart
  * Allow at the same checkpoint given the code change does not have conflict to the checkpoint (e.g. different aggregation), it will throw exception
  * Need to perform impact analysis before code change regardless whether spark allows it.  * Need
    to perform impact analysis before code change regardless whether spark allows it.
  
  

## Reference:

* https://github.com/LearningJournal/Spark-Streaming-In-Python