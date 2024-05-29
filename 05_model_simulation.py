# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2024
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

USERNAME = os.environ["PROJECT_USER"]
DBNAME = "LOGISTICS_MLOPS_{}".format(USERNAME)
CONNECTION_NAME = "go01-aw-dl"
TODAY = datetime.date.today()

# Instantiate BankDataGen class
dg = IotDataGen(USERNAME, DBNAME, CONNECTION_NAME)

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
apiUtil = ApiUtility(project_id, USERNAME)
model_name = f"TimeSeriesQuery-{USERNAME}-2024-05-29" # Update model name here

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

percent_counter = 0

for i in range(1000):
  print("Added {} records".format(percent_counter)) if (
      percent_counter % 25 == 0
  ) else None
  percent_counter += 1
  submitQuery(Model_AccessKey)

# You can use an APIV2-based utility to access the latest model's metadata. For more, explore the src folder
apiUtil = ApiUtility(project_id, USERNAME)
model_name = f"MultiDimMotif-{USERNAME}-2024-05-29" # Update model name here

Model_AccessKey = apiUtil.get_latest_deployment_details(model_name=model_name)["model_access_key"]
Deployment_CRN = apiUtil.get_latest_deployment_details(model_name=model_name)["latest_deployment_crn"]

def submitMotifSearch(df, Model_AccessKey):
    """
    Method to create and send a synthetic request to Multidimensional Motif Discovery Model
    """
    df = df[["iot_signal_1", "iot_signal_2", "iot_signal_3", "iot_signal_4"]]
    record = df.to_dict(orient='index')
    response = cdsw.call_model(Model_AccessKey, record)

    return response


for i in range(10):
    # Instantiate BankDataGen class
    dg = IotDataGen(USERNAME, DBNAME, CONNECTION_NAME)

    # Create CML Spark Connection
    spark = dg.createSparkConnection()

    # Create IoT Fleet DF
    df_desmoines = dg.dataGen(spark)
    df_desmoines = dg.addCorrelatedColumn(df_desmoines)
    df = df_desmoines.toPandas()

    print("Added {} records".format(percent_counter)) if (percent_counter % 25 == 0) else None
    percent_counter += 1
    submitMotifSearch(df, Model_AccessKey)
