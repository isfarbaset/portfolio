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

#Changing it for a dumb reasons
df_in = spark.read.parquet(args.input_data)
logging.info(f"finished reading files...")

state_names = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]

state_names.append("sentiment_num")
state_names.append("rawFeatures")

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
idf = VectorAssembler(inputCols= state_names, outputCol="features")

rf = RandomForestRegressor(featuresCol="features", labelCol = "score")

pipeline_forest = Pipeline(stages=[tokenizer, hashingTF, idf, rf])

(trainingData, testData) = df_in.randomSplit([0.7, 0.3])

model_forest = pipeline_forest.fit(trainingData)
predictions_forest = model_forest.transform(testData)

datastore = 'azureml://datastores/workspaceblobstore/paths'
model_forest.write().overwrite().save(f"{datastore}/models/ml_model_forest_score_prediction")

evaluator = RegressionEvaluator(labelCol="score", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions_forest)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

spark.stop()



