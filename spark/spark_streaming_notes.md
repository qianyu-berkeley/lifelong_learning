# Spark Real-time Stream Processing

## Spark Structured Streaming Intro

* Batch processing is a subset of streaming processing which extends spark batch processing
* Stream processing solve batch processing problems
  * Check points
  * Tracking last processed files
  * fault tolarance
* Unsupported operations
  * sorting `orderBy`
  * `limit`
  * `distinct`
  * `count()`
  * `foreach()`

## Spark Streaming framework

* Micro-batch
  * Autolooping between micro-batches
  * batch start, end position management
  * Intermediate state management
  * combine results to the prevous batch results
  * fault tolarance and restart management
    * End-to-end exactly-once fault tolerance guarantees through checkpointing and write ahead logs
      * record the offset range of data being processed during each trigger interval
      * **idempotent** multiple writes of the same data do not result in duplicates being written to the sink
    * Requirement:
      * Streaming source is replayable -- cloud-based object storrage and pub/sub messaging services

## Spark Structured Streaming API

* Dataframe based streaming API
* SQL Engine optimization
* Support event time semantic
* Future enhancement and features
* 3 Step processes
  * Read streaming source  - input dataframe
  * Data Transform - process and transform input dataframe to output dataframe
  * Write to output - streaming sink
* `readStream` (returns a `DataStreamReader`)
  * take options of input source (e.g. `socket`, `kafka`, `json`) use option of `host`, `port`, `path`)
  * Can perform operations of standard spark dataframe
  * Config a streaming read on a source requires:
    * The schema of the data (it is not safe to infer schema, since we assume the source is grow indfinitely from zero)
      * For pub/hub system like `kafka` or `event hub`, the source will provide schema
    * The `format` of the source (`file format or named connector`)
    * Config specific to the source
      * `kafka`
      * `Event hubs`
* `writeStream` (returns a `DataStreamwriter`) take mandortary options of `checkpointLocation` and outputMode (e.g. `complete`)
  * The `format` of output sink (e.g. `parquet`, `kafka`)
  * The location of the **checkpoint directory** (Note: checkpoint cannot be shared between seperate streams)
  * [The output mode](#output-mode)
  * `start()` or `.start(filepath)` to trigger the job
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
    * Example:

      ```python
      streamingDF = spark.readStream \
                  .format("json") \
                  .schema(schema) \
                  # Optional; force processing of only 1 file per trigger 
                  .option("maxFilesPerTrigger", 1)  \
                  .load(dataPath)
      ```

  * Kafka source (or other pub/hub system)
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
* `Update`: **upsert** like operation i.e. only the rows in the result table that were updated since the last trigger will be outputted to the sink
* `Compele`: **overwrite** the complete results

## Trigger Interval

* triggers are specified when defining how data will be written to a sink and control the frequency of micro-batches.
* If undefined (default): The query will be executed as soon as the system has completed processing the previous query
* Fixed Interval micro-batches: `.trigger(Trigger.ProcessingTime("2 minutes"))` The query will be executed in micro-batches and kicked off at the user-specified intervals
* One-time micro-batch: `.trigger(Trigger.Once())` The query will execute only one micro-batch to process all the available data and then stop on its own
  * streaming trigger runOnce is better than batch job, why?
    * Bookkeeping: structured streaming does low-level bookkeeping
    * Table level atomicity (fault-tolerance: Structured Streaming commits all files created by the job to a log after each successful trigger. When Spark reads back the table, it uses this log to figure out which files are valid.)
    * Stateful operations across runs: With Structured Streaming, it’s as easy as setting a watermark and using dropDuplicates(). By configuring the watermark long enough to encompass several runs of your streaming job, you will make sure that you don’t get duplicate data across runs.
* Continous w/fixed checkpoint interval: `.trigger(Trigger.Continuous("1 second"))` The query will be executed in a low-latency

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
  
## Time in streaming

* Multiple times definitions in respect to data
  * **when data is Generated** => `event time`
  * when data is Written to the straaming source
  * when data is Processed into Spark  
* Most analytics will be interested in the time the data was generated

## Windowing and Aggregates


* Stateless vs. Stateful transformations
  * stateless: `select()`, `filter()`, `map()`, `flatMap()`, `explode()`
    * complete output mode is not supported
  * stateful: `grouping`, `aggregation`, `windowing`, `joins` 
    * need to maintain states across micro-batch
    * excessive state causes out of memory
    * spark support 2 stateful operations:
      * Managed stateful operation (spark manage the clean-up)
        * Time-bound aggregation is a good candidate for spark to manage the clean up
      * unmanaged stateful operations (only allowed in java and scala)
        * continuous aggregation need to have a user defined clean-up
* Window aggregates
  * Defining windows on a time series field allows users to utilize this field for aggregations in the same way they would use distinct values when calling `GROUP BY`. 
  * Aggregation window has nothing to do with trigger time (which is the time we start processing a micro-batch)
  * time window is nothing but an aggregation column
  * The state table will maintain aggregates for each user-defined bucket of time. Spark supports two types of windows:
    * `Tumbling Time` window: a series of fixed size, **non-overlapping** windows
      * each event can be part of only 1 window
      * if a record is late, spark would use the saved state information to recompute and update the right aggregation window with the record based on event time

      ```python
      window_agg_df = df \
        .groupBy(window(col("createdTime"), "15 minute")) \
        .agg(sum("Buy").alias("Totalbuy"), 
            sum("Sell").alias("Totalsell"))
      ```

      * limitation: cannot perform running total type of analytical aggregations. The solution is to create a seperate batch processing to perform those type of transformations.

    * sliding time window (**overlapping**)
      * to compute moving aggregates
      * each events can be part of multiple sliding window

      ```python
      window_agg_df = df \
        .groupBy(window(col("createdTime"), "15 minute", "5 minute")) \
        .agg(sum("Buy").alias("Totalbuy"), 
            sum("Sell").alias("Totalsell"))
      ```

  The diagram below from the <a href="https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html" target="_blank">Structured Streaming Programming Guide</a> guide shows sliding windows.
  <img src="http://spark.apache.org/docs/latest/img/structured-streaming-window.png">
  
  * Performance Consideration
    * Because aggregation will trigger a shuffle, configuring the number of partitions can reduce the number of tasks and properly balance the workload for the cluster.
    * In most cases, a 1-to-1 mapping of partitions to cores is ideal for streaming applications. The code below sets the number of partitions to 8, which maps perfectly to a cluster with 8 cores.

* Watermark
  * a watermark is the expiration time for saved states, a key for state store cleanup
  * events within the watermark is taken, event outside may or may not be taken
  * How to set watermark? ask
    * What is the maximum possible delay?
    * When late records are not relevant?
      e.g. we want accuracy >= 99.99%, we don't care records after 30 mins

    ```python
    # Example one
    # define watermark before the groupby, use the same time column of groupby
    window_agg_df = df \
      .withWatermark("CreatedTime", "30 minute") \
      .groupBy(window(col("createdTime"), "15 minute")) \
      .agg(sum("Buy").alias("Totalbuy"), 
           sum("Sell").alias("Totalsell"))
    
    # Example 2
    watermarkedDF = (streamingDF
      .withWatermark("time", "2 hours")           # Specify a 2-hour watermark
      .groupBy(col("action"),                     # Aggregate by action...
               window(col("time"), "1 hour"))     # ...then by a 1 hour window
      .count()                                    # For each aggregate, produce a count
      .select(col("window.start").alias("start"), # Elevate field to column
              col("action"),                      # Include count
              col("count"))                       # Include action
      .orderBy(col("start"), col("action"))       # Sort by the start time
    ```

    * Max(Event Time) - Watermark = Watermark Boundry => State earlier than watermark boundry will be cleaned.
    * Output modes:
      * if set `complete` mode, spark will try to give complete output therefore does not use watermark for cleaning
      * if set `update` mode, spark will use water mark to clean the state, it is **most useful** output mode for streaming aggregation. But do not use this mode for append only sinks (e.g. file sink), it will create duplicated records. Use with sinks support upsert operations.
      * `append` mode can work with watermark, spark will suppress the output of the window aggregates until it pass the watermark boundry to maintain record (We can use with file sink with delay)

## Spark Streaming Join

* Streaming Dataframe to static dataframe (stream enrichment)
  * Stateless
  * no watermark or windowing needs to be configured, and distinct keys from the join accumulate over time. Each streaming microbatch joins with the most current version of the static table.
  * Approach:
    * Create streaming dataframe from kafka
    * Create static dataframe from the database table (e.g. cassandra)
    * perform streaming join
    * write back to the database (e.g. cassandra)
  * Need to set:
    * connection to database either using option or set in config
  * There's nothing especially complicated about the syntax used for a stream-static join. The primary thing to keep in mind is that our streaming table is driving the action. For each new batch of data we see arriving in streaming table, we'll process our join logic.
  
  Example 1
  ```python
  silverDF = spark.readStream.table("silver_recordings")
  piiDF = spark.table("pii")
  joinedDF = silverDF.join(piiDF, on=["mrn"])
  joinedDF.writeStream \
    .trigger(processingTime="5 seconds") \
    .option("checkpointLocation", enrichedCheckpoint) \
    .toTable("enriched_recordings")
  ```

  Example 2
  ```python
  join_expr = login_df.login_id == user_df.login_id
  join_type = "inner"
  joined_df = login_df.join(user_df, join_expr, join_type) \
    .drop(login_df.login_id)
  output_df = joined_df.select(col("login_id"), col("user_name"),
                               col("created_time").alias("last_login"))

  # use following appraoch to sink source if there is no predefined sink
  # we define a custom function "write_to_cassandra"
  output_query = output_df.writeStream \
    .foreachBatch(write_to_cassandra) \
    .otuputMode("update") \
    .option("checkpointLocation", "chk-pint-dir") \
    .option(processingTime="1 minute") \
    .start()
    
  def write_to_cassandra(target_df, batch_id):
    target_df.write \
      .format("org.apache.spark.sql.cassandra") \
      .option("keyspace", "spark_db")
      .option("table", "users") \
      .mode("append") \
      .save()
    target_df.show()
  ```

* stream to stream join 
  * Stateful
    * Records (from both streaming df) is kept in stateful store to ensure one to many join
    * spark does not know when to clean the states
    * duplicate event can cause incorrect results, it is up to user to ensure it does not happen
  * Approach:
    * Reading from 1st kafka topic to a streaming dataframe
    * Reading from 2nd kafka topic to a streaming dataframe
    * perform streaming join
    
    ```python
    join_expr = "ImpressionID == ClickID"
    join_type = "inner"
    
    joined_df = impressions_df.join(clicks_df, expr(join_expr), join_type)
    output_query = joined-df.writeStream \
      .format("console") \
      .outputMode("append")
      .option("checkpointLocation", "chk-pint-dir") \
      .option(processingTime="1 minute") \
      .start()
    ```

  * Add watermark to clean up stateful store when generate the streaming dataframe

  ```python
  impressions_df = kafka_impression_df \
    .select(from_json(col("value").cast("string"), impressionSchema).alias("value")) \
    .selectExpr("value.ImpressionID", "value.CreatedTime", "value.Campaigner") \
    .withColumn("ImpressionTime", to_timestamp(col("CreatedTime"), "yyyy-MM-dd HH:mm::ss")) \
    .drop("CreatedTime") \
    .withWatermark("ImpressionTime", "30 minute")
  ```
  
   clean up state store after 30 mins

  * Spark does not garanttee that records outside of water mark will be 100 ignore but spark will garaunttee that the records inside the watermark will be available

* Streaming outer join
  * streaming outer join with a streaming datafram and static dataframe
    * Left outer: left side must be a stream
    * Right outer: Right side must be a stream
  * streaming outer join between 2 streaming dataframe
    * watermarket is mandatary to ensure both all outer join works
    * left outer
      * watermark on the right-side stream
      * Max time range constraint between and left and right-side events
    * right outer
      * watermark on the left-side stream
      * Max time range constraint between and left and right-side events

  ```python
  join_expr = "ImpressionID == ClickID" + \
            " AND ClickTime Between ImpressionTime AND ImpressionTime + interval 15 minute"
  ```

## Databricks Autoloader

* Recommended method for streaming raw data from cloud object storage
* For small datasets, the default directory listing execution mode will provide provide exceptional performance and cost savings.
* As the size of your data scales, the preferred execution method is file notification, which requires configuring a connection to your storage queue service and event notification, which will allow Databricks to idempotently track and process all files as they arrive in your storage account.
* Configuring Auto Loader requires using the `cloudFiles` format.

  ```python
  stream_job = spark.readStream \
      .format("cloudFiles") \
      .option("cloudFiles.format", "json") \
      .schema(schema) \
      .load(raw) \
      .writeStream \
      .format("delta") \
      .option("checkpointLocation", gymMacLogsCheckpoint) \
      .trigger(once=True) \
      .start("/temp/mac_log/") \
      .awaitTermination()
  display(spark.sql(f"DESCRIBE HISTORY delta.`/temp/mac_log`))
  ```

* auto loader with trigger once logic prevents any CDC (Change Data Capture) on our file system, allowing us to simple trigger a chron job daily to process all new data


## Tools

* Users should define a `StreamingQueryListener`, as demonstrated [here](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#reporting-metrics-programmatically-using-asynchronous-apis).
* The `StreamingQuery` object can be used to [monitor and manage the stream](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#managing-streaming-queries).
* The `StreamingQuery` object can be captured as the return of a write definition or accessed from within the active streams list, demonstrated here:

  ```python
  for s in spark.streams.active:         # Iterate over all streams
    print(s.id)    
  print(streamingQuery.recentProgress) # access metadata about recently processed micro-batches
  ```

* Stop streaming Query

  ```python
  streamingQuery.awaitTermination(5)      # Stream for another 5 seconds while the current thread blocks
  streamingQuery.stop()                   # Stop the stream
  ```

* Within the Databricks notebooks, we can use the `display()` function to render a live plot. This stream is written to memory; **generally speaking this is most useful for debugging purposes**.
* When you pass a "streaming" `DataFrame` to `display()`, you trigger a streaming job:
  * A "memory" sink is being used
  * The output mode is complete
  * *OPTIONAL* - The query name is specified with the `streamName` parameter
  * *OPTIONAL* - The trigger is specified with the `trigger` parameter
  * *OPTIONAL* - The checkpointing location is specified with the `checkpointLocation`
  * `display(myDF, streamName = "myQuery")`

* Since the `streamName` gets registered as a temporary table pointing to the memory sink, we can use SQL to query the sink.
* To stop all remaining streams

  ```python
  for s in spark.streams.active:
    s.stop()
  ```

## Reference:

* https://github.com/LearningJournal/Spark-Streaming-In-Python
* CDC: Change data capture (CDC) refers to the process of identifying and capturing changes made to data in a database and then delivering those changes in real-time to a downstream process or system)