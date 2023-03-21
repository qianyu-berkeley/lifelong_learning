# Jenkins Notes

## Introduction

* Top DevOps CI/CD Tool
* Automate builds and dev
* Open source with 1400 plugins
* Industry use declarative pipelines

## Setup

* Easy Setup with docker image based on https://github.com/jenkinsci/docker
  * Key: `c72c3e2444d4455da6503697ef9d6a88`

## Jenkins 101

* Commonly used
  * Freestyle project
  * Pipeline
  * Multibranch pipeline enable different github branch builds

* Declarative way to setup Jenkins
  * Create a Jenkinsfile in the project github repo
  * Setup a pipeline with pipeline script

* Build trigger
  * Build periodically: e.g. setup in the morning, tear down in the evening
  * Poll SCM:  based on github changes

## Build a pipeline

* Reference:
  * https://www.jenkins.io/doc/book/pipeline/syntax/
  * https://www.jenkins.io/doc/pipeline/steps/workflow-basic-steps/

* pipeline script
  * start with `pipeline{ ... }` as the top level
  * `agent` define which node the pipeline is running
  * `stages` add every stages (green blocks in the pipeline view)
  * `steps`  steps within stage
  * `dir()` same is `cd` of shell command to ensure every stage is go to the home folder

* Debugger tool: (not all company enable it)
  * `replay`
* pipeline script from scm
* build Trigger with `Poll SCM`: set 2 mins as general practices with Cron syntax `H/2 * * * *` 

## Multi-branch

* Build configuration by Jenkinsfile
* Scan Multibranch Pipeline Triggers (2 mins)

## Parameterized pipelines

* Boolean Parameter
* Input Parameters (ref: https://github.com/jenkinsci/pipeline-model-definition-plugin/wiki/Parametrized-pipelines)
  * String
  * Text (support multi lines)
* we can check parameter menu to see which param is used


## Variables

* We can define our own
* We normally use Jenkins default env varibles (ref: https://www.jenkins.io/doc/book/pipeline/jenkinsfile/#using-environment-variables)

## Advanced 

* Groovy: the language allow more complex functionalities
  * used in `script {}` block such as `if` statement
* Troubleshooting:
  * Replay
  * Console output
  * stack overflow
* debug trick (sleep function)
* functions
  * outside of `pipeline{}` block with `def {}`
  * variable inside of `pipeline{}` block cannot be accessed by `def {}` block since it is outside the scope
* Variable scope
  * To make variable available in all scope, use `environment {}` block which make them global
* multi-line bash shell use `""" """`

## Reference:

[Jenkins training examples](https://github.com/addamstj/Jenkins-Course.git)