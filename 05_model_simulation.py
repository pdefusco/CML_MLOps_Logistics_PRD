# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

## Part 7a - Model Operations - Drift Simulation
#
# This script showcases how to use the model operations features of CML.
# # This feature allows machine learning engineering to **measure and manage models
# through their life cycle**, and know how a model is performing over time. As part
# of the larger machine learning lifecycle, this closes the loop on managing
# models that have been deployed into production.

### Add Model Metrics
# New  metrics can be added to a model and existing ones updated using the `cdsw`
# library and the [model metrics SDK](https://docs.cloudera.com/machine-learning/cloud/model-metrics/topics/ml-tracking-model-metrics-using-python.html)
# If model metrics is enabled for a model, then every call to that model is recorded
# in the model metric database. There are situations in which its necessary to update or
# add to those recordered metrics. This script shows you how this works.

#### Update Exsiting Tracked Metrics
# This is part of what is called "ground truth". Certain machine learning implemetations,
# (like this very project) will use a supervised approach where a model is making a
# prediction and the acutal value (or label) is only available at a later stage. To check
# how well a model is performing, these actual values need to be compared with the
# prediction from the model. Each time a model endpoint is called, it provides the response
# from the function, some other details, and a unique uuid for that response.
# This tracked model response entry can then be updated at a later date to add the
# actual "ground truth" value, or any other data that you want to add.
#
# Data can be added to a tracked model response using the `cdsw.track_delayed_metrics`.
#
# ```python
# help(cdsw.track_delayed_metrics)
# Help on function track_delayed_metrics in module cdsw:
#
# track_delayed_metrics(metrics, prediction_uuid)
#    Description
#    -----------
#
#    Track a metric for a model prediction that is only known after prediction time.
#    For example, for a model that makes a binary or categorical prediction, the actual
#    correctness of the prediction is not known at prediction time. This function can be
#    used to retroactively to track a prediction's correctness later, when ground truth
#    is available
#        Example:
#            >>>track_delayed_metrics({"ground_truth": "value"}, "prediction_uuid")
#
#    Parameters
#    ----------
#    metrics: object
#        metrics object
#    prediction_uuid: string, UUID
#        prediction UUID of model metrics
# ```

#### Adding Additional Metrics
# It is also possible to add additional data/metrics to the model database to track
# things like aggrerate metrics that aren't associated with the one particular response.
# This can be done using the `cdsw.track_aggregate_metrics` function.

# ```python
# help(cdsw.track_aggregate_metrics)
# Help on function track_aggregate_metrics in module cdsw:
#
# track_aggregate_metrics(metrics, start_timestamp_ms, end_timestamp_ms, model_deployment_crn=None)
#    Description
#    -----------
#
#    Track aggregate metric data for model deployment or model build or model
#        Example:
#            >>>track_aggregate_metrics({"val_count": 125}, 1585685142786,
#            ... 1585685153602, model_deployment_crn="/db401b6a-4b26-4c8f-8ea6-a1b09b93db88"))
#
#    Parameters
#    ----------
#    metrics: object
#        metrics data object
#    start_timestamp_ms: int
#        aggregated metrics start timestamp in milliseconds
#    end_timestamp_ms: int
#        aggregated metrics end timestamp in milliseconds
#    model_deployment_crn: string
#       model deployment Crn
# ```
#

### Model Drift Simlation
# This script simulates making calls to the model using sample data, and slowly
# introducting an increasing amount of random variation to the churn value so that
# the model will be less accurate over time.

# The script will grab 1000 random samples from the data set and simulate 1000
# predictions. The live model will be called each time in the loop and while the
# `churn_error` function adds an increasing amount of error to the data to make
# the model less accurate. The actual value, the response value, and the uuid are
# added to an array.
#
# Then there is "ground truth" loop that iterates though the array and updates the
# recorded metric to add the actual lable value using the uuid. At the same time, the
# model accruacy is evaluated every 100 samples and added as an aggregate metric.
# Overtime this accuracy metric falls due the error introduced into the data.

import cdsw
import time, os, random, json, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmlbootstrap import CMLBootstrap
from pyspark.sql import SparkSession
import cmlapi
from src.api import ApiUtility
import cml.data_v1 as cmldata
from utils import IotDataGen
import datetime

#---------------------------------------------------
#               CREATE BATCH DATA
#---------------------------------------------------

USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "LOGISTICS_MLOPS_DEMO"
STORAGE = "s3a://goes-se-sandbox01"
CONNECTION_NAME = "se-aw-mdl"
TODAY = datetime.date.today()

# Instantiate BankDataGen class
dg = IotDataGen(USERNAME, STORAGE, DBNAME, CONNECTION_NAME)

# Create CML Spark Connection
spark = dg.createSparkConnection()

# Create IoT Fleet DF
df_desmoines = dg.dataGen(spark)
df_desmoines = dg.addCorrelatedColumn(df_desmoines)
df = df_desmoines.toPandas()

# You can access all models with API V2
client = cmlapi.default_client()
project_id = os.environ["CDSW_PROJECT_ID"]
client.list_models(project_id)

# You can use an APIV2-based utility to access the latest model's metadata. For more, explore the src folder
apiUtil = ApiUtility()
model_name = f"TimeSeriesQuery-{USERNAME}-{TODAY}" # Update model name here

Model_AccessKey = apiUtil.get_latest_deployment_details(model_name=model_name)["model_access_key"]
Deployment_CRN = apiUtil.get_latest_deployment_details(model_name=model_name)["latest_deployment_crn"]

import random
import numpy as np

def submitQuery(Model_AccessKey):
    """
    Method to create and send a synthetic request to Time Series Query Model
    """

    randomInts = [random.randint(50,54) for i in range(4)]
    record = '{"pattern": ""}'
    data = json.loads(record)
    data["pattern"] = randomInts
    response = cdsw.call_model(Model_AccessKey, data)

    return response

for i in range(10):
  submitQuery()

# You can use an APIV2-based utility to access the latest model's metadata. For more, explore the src folder
apiUtil = ApiUtility()
model_name = f"MultiDimMotif-{USERNAME}-{TODAY}" # Update model name here

Model_AccessKey = apiUtil.get_latest_deployment_details(model_name=model_name)["model_access_key"]
Deployment_CRN = apiUtil.get_latest_deployment_details(model_name=model_name)["latest_deployment_crn"]

def submitMotifSearch(df, Model_AccessKey):
    """
    Method to create and send a synthetic request to Multidimensional Motif Discovery Model
    """

    record = iotDf.to_json(orient="columns")
    response = cdsw.call_model(Model_AccessKey, record)

    return response

submitMotifSearch(df, Model_AccessKey

# Get the various Model Endpoint details
HOST = os.getenv("CDSW_API_URL").split(":")[0] + "://" + os.getenv("CDSW_DOMAIN")
model_endpoint = (
    HOST.split("//")[0] + "//modelservice." + HOST.split("//")[1] + "/model"
)
