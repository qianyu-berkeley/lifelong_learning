# Databricks Platform and Knowledge

## Databricks Lakehouse

### Problem with tranditional datalake

* Lack of ACID transaction support
* Lack of schema enforcement
* Lack of integration of data catalog
* Ineffective partitioning
* Too many small files (cause poor query performance)

### Lakehouse Technologies

### Deltalake

* A file based open source technology
* Benefits
  * Guarantee ACID transation
  * Scalable data and metadata handling for large datatables
  * Audit history and time travel
  * Schema enforcement and schema evolution
  * Support deletes, updates, and merges
  * To accomadate complex use cases: e.g. CDC (Change data capture) SCD (Slowly change dimension operation), streaming upsert
  * Unified streaming and batch data processing
* Run on top of existing datalake and compatible with spark
* Use delta table based on Apache parquet (can switch from parquet table to delta lake table easy and quick)
* Has a transaction log
  * When query or read table, spark check the transaction log for new transactions, and update the table based on new transations
  * Prevent divergent and conflict changes to the table
* It is open source to enable flexibility
* Integration with all public analytic tools

### Photon

* It is a query engine to support fast, efficient, and cost saving performance
* Compatible with Spark for both batch and streaming data
  * SQL based jobs
  * IoT Use case
  * Data privacy and compliance
  * loading data into Delta and parquet
  
### Unified Governance and Security

* To solve problems of:
  * Diversity of data and AI assets
  * imcompatible and disparate data platforms
  * mult-cloud adoption
  * fragmented tool usage
* Key components:
  * Unity catalog: 
    * a unified governance model based on ANSI SQL
    * Access controls
    * User management
    * Metastore
    * Allow access control at rows and cols to user groups
    * Attributes based access control to enable governance at scale
    * Provde a detailed audit trail
    * Fast metadata processing 
    * Autmoated e2e data lineage (visual) down to table
    * Support delta sharing natively
  * Delta sharing for live data
    * Open cross-platform sharing tool
    * Share live data without copying it
    * Centroalized admin and governance
    * Marketplace for data products
    * Provacy-safe data clean rooms
  * Control plane and data plane: enable security
    * Control plane
      * Web app, config, notebook, repo, cluster manager
    * Data plane
      * Cluster, customer storage
      * encrypted data-at-rest
      * Serverless data plane networking infra is managed by databricks
* User Identity and access
  * Table ACLs (access control list) feature
  * IAM instance profiles
  * Securely stored access key
  * The secrets API
* Data Security
  * Databricks encryption capabilities are in place both at rest and in motion
  * data-at-rest encryption
    * control plane is encrypted
    * data plan supports local encryption
    * customer can use encrypted storage buckets
    * customer at some tiers can config customer-managed keys for managed service
  * data-in-motion encryption
    * control plane <=> data plane is encrypted
    * offers optional intra-cluster encryption
    * customer role can be written to avoid unencrypted service (e.g. FTP)
    
### Instance compute and serverless

* Serverless data plane
  * Databricks serverless SQL
  * serverless SQL compute managed by databricks, Elastic and Secure
  
### Lakehouse Data Management Terminology (in the data object hierachy)

* Metastore
  * Top level logical container in unit catalog, it is a logical construct that represents the metadata for organizing your data
  * Metadata is the information about the data objects being managed by the metastore
  * In improves upone Hive metastore which is a local metastore linked to each databricks workspace on security and auditing capability
  * Store in the control plane, (data managed by the metastore is stored in the cloud storage container)
* Catalog
  * Top most conatainer for data objects in unit catalog
  * A Metastore can have many catalog
  * Form the first part of three-level namespace that data analysts use to reference data objects in unit catalog
    * `select * from catalog.schema.table`
* Schema
  * Part of traditional SQL, unchanged, function as a container for table, view and functions
* Table
  * metadata: comments, tags, datatypes managed by the control plane
  * managed table (data file in managed stores location)
  * external table (data files are stored in an external stored location)
* View
  * stored queries executed
  * Perform arbitrary SQL transformation on tables and other views
  * Read-only
  * Cannot modify the underlying data
* Function
  * custom function that can be envoked within queries

### Databricks Lakehouse Workload

* Data warehouse
  * Performance
  * Build-in governance
  * Rich ecosystem
  * Keep single copy in existing data lakes
  * Integrated with Unit catalog
* Data Engineering  
  * Tasks: Ingestion => Transformation => Orchastration
  * Challenges for the data engineering workloads:
    * complex data ingestion methods
    * support data engineering principle
    * Third-part orchastration tools
    * pipeline and architecture performance tuning
    * Inconsistencies between data warehouse and data lake providers
  * How Lakehouse solve the challenge? It provides a **unified data platform** with **managed data ingestion**, schema detection, enforcement, and evolution, paired with **declarative, auto-scaling data flow** integrated with a lakehouse **native orchestrator** which supports all kinds of workflow. It provides E2E solutions for ingesting, transforming, processing, scheduling and delivering data
    * Capabilities
      * Easy data ingestion
      * Automated ETL pipelines
      * Data quality checks
      * Batch and streaming tuning
      * Automatic recovery
      * Data pipeline observability
      * Simplified operations
      * scheduling and orchastration
    * Tools 
      * Ingestion
        * `Auto Loader`: auto detect schema and enforces it
      * Transformation
        * `Delta Live Tables` (DLT): apply declarative approach to build data pipelines and auto scales infrastructure, Support batch and streaming with the same API
      * Orchestrate
        * `workflow`: Allow user to orchestrate data flow piplines written in DLT or dbt, Machine learing pipeline or notebook. Use can also create workflow using its API with 3rd party tools such as airflow
* Data Streaming
  * Use cases
    * Real-time Analysis
    * Real-time Machine learning
    * Real-time application
  * Capabilities
    * Fast development
    * Simplify operations with automation
    * Unified governance for real time and historical data
* Machine Learning / Data Science
  * Compute platform to support any ML workload and ML run-time
    * ML framework build-in
    * Build-in support for distributed training
    * AutoML
    * Hardware accelerators

## Databricks Clusters

* All-purpuse cluster: for interactive notebook
  * Can be created manually, use cli, Rest API
  * Can be terminated manually & Restart
* Job cluster: run fast, robust automated jobs
  * create by databricks job scheduler
  * cannot restart a job cluster
  * terminated at the end of the job

## Reference

### Data Terminologies

* ACID Transation: In the context of transaction processing, the acronym ACID refers to the four key properties of a transaction: atomicity, consistency, isolation, and durability. All changes to data are performed as if they are a single operation. That is, all the changes are performed, or none of them are.