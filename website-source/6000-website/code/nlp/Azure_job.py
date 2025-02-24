import argparse
import logging
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import lower


logging.basicConfig(level=logging.INFO)

# Parse Inputs
parser = argparse.ArgumentParser()
parser.add_argument("--input_object_store_base_url")
parser.add_argument("--input_path")
parser.add_argument("--output_object_store_base_url")
parser.add_argument("--output_path")
args = parser.parse_args()

logging.info(args.input_object_store_base_url)
logging.info(args.input_path)
logging.info(args.output_object_store_base_url)
logging.info(args.output_path)

input_complete_path = f"{args.input_object_store_base_url}{args.input_path}"
output_complete_path = f"{args.output_object_store_base_url}{args.output_path}"

logging.info(input_complete_path)
logging.info(output_complete_path)

spark = SparkSession.builder.appName("PySparkApp").getOrCreate()
logging.info(f"spark version = {spark.version}")

# Read data from object store
logging.info(f"going to read {input_complete_path}")
df_in = spark.read.parquet(input_complete_path)
df_in_ct = df_in.count()
logging.info(f"finished reading files...")

# filter the dataframe to only keep the subreddits of interest
comments_cols = ["author", "body", "controversiality", "created_utc", "distinguished", "edited", "gilded", "id",
                 "parent_id", "score", "subreddit", "subreddit_id"]
state_names = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
state_names = [x.lower() for x in state_names]
state_regex = "|".join(state_names)
df = df_in.select(comments_cols)
df_filtered = df.filter(lower(col("body")).rlike(state_regex))
filtered_ct = df_filtered.count()

# save the filtered dataframes so that these files can now be used for future analysis
logging.info(f"going to write {output_complete_path}")

logging.info(f"Read in {df_in_ct} records, wrote out {filtered_ct} records.")
df_filtered.write.mode("overwrite").parquet(output_complete_path, compression="zstd")

spark.stop()
