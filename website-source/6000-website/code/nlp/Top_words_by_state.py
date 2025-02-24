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

from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import explode, col

state_names = ["American Samoa", "Alaska", "Alabama", "Arkansas", "Arizona", "California", 
"Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Guam", 
"Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", 
"Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", 
"Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", 
"New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", 
"Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", 
"Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
state_names = [x.lower() for x in state_names]

#Changing it for a dumb reasons
df_in = spark.read.parquet(args.input_data)
import pandas as pd
topword_list = []
for i in range(0,len(state_names)):
    logging.info(f"{state_names[i]}")
    df_subset = df_in.filter(df_in[state_names[i]] == True)
    tokenizer = Tokenizer(inputCol="text", outputCol="words_token")
    tokenized = tokenizer.transform(df_in).select('words_token')
    stopwordList = [" ", "", "|", "like", "-", "get"] 
    stopwordList.extend(StopWordsRemover().getStopWords())
    stopwordList = list(set(stopwordList))#optionnal
    StopWordsRemover(inputCol="words", outputCol="filtered")

    remover = StopWordsRemover(inputCol='words_token', outputCol='words_clean', stopWords=stopwordList)
    data_clean = remover.transform(tokenized).select('words_clean')

    result = data_clean.withColumn('word', explode(col('words_clean'))) \
        .groupBy('word') \
        .count().sort('count', ascending=False)
    val = result.select("word").limit(1).collect()[0][0]
    logging.info(f"{val}")

    topword_list.append((state_names[i], val))

df_output =  pd.DataFrame(topword_list, columns=['State', 'top_word'])
df_out = spark.createDataFrame(df_output)
df_out.coalesce(1).write.mode("overwrite").csv("azureml://datastores/workspaceblobstore/paths/states/state_level_topword.csv")

spark.stop()



