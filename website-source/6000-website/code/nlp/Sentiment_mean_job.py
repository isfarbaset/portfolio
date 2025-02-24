import argparse
import logging
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
import sparknlp
import pyspark
from sparknlp.annotator import *
from sparknlp.pretrained import *
from azureml.core import Workspace, Dataset, Datastore
from pyspark.sql.functions import explode

logging.basicConfig(level=logging.INFO)

import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

print("Spark NLP version: ", sparknlp.version())

## You need to add the spark-nlp jar to the spark session

spark = SparkSession.builder \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.0.2") \
    .getOrCreate()

print(f"Apache Spark version: {spark.version}")

#spark.sparkContext.getConf().getAll()

import sys

from pip import _internal

print("Printing Spark COnfiguration")
#print(spark.sparkContext.getConf().getAll())

parser = argparse.ArgumentParser()
parser.add_argument("--input_data")

args = parser.parse_args()

# Read data from object store

df_in = spark.read.parquet(args.input_data)
logging.info(f"finished reading files...")

pipeline = PretrainedPipeline("analyze_sentimentdl_use_twitter", lang="en")

state_names = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
state_names = [x.lower() for x in state_names]


results = pipeline.transform(df_in.withColumnRenamed("body", "text"))
#results = df_in

for i in range(len(state_names)):
    state_name = state_names[i]
    logging.info("For loop for " + state_name)
    results = results.withColumn(state_names[i], when(lower(col("text")).contains(state_names[i]), True).otherwise(False))

results = results.withColumn("sentiment2", explode("sentiment"))
results = results.withColumn("sentiment_num", when(col("sentiment2.result") == "positive", 1).when(col("sentiment2.result") == "neutral", 0).otherwise(-1))
results.select("sentiment_num").show(truncate = False)
results = results.drop("sentiment")
results = results.drop("sentiment2")
results = results.drop("document")
results = results.drop("sentence_embeddings")

datastore = 'azureml://datastores/workspaceblobstore/paths'
results.write.mode('overwrite').parquet(f"{datastore}/states/states_sentiment/", compression="zstd")


for i in range(len(state_names)):
    state_inst = results.filter(results[state_names[i]] == True).agg(mean(col("sentiment_num")))
    val = state_inst.collect()[0][0]
    avg_list.append((state_names[i], val))

df_output =  pd.DataFrame(avg_list, columns=['State', 'Avg_sentiment'])
df_out = spark.createDataFrame(df_output)
df_out.write.mode("overwrite").csv("azureml://datastores/workspaceblobstore/paths/states/state_level_sentiment.csv")

spark.stop()



