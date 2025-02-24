
import os
import logging
import argparse

# Import pyspark and build Spark session
from pyspark.sql.functions import *
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

logging.basicConfig(format='%(asctime)s,%(levelname)s,%(module)s,%(filename)s,%(lineno)d,%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def main():
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_dataset_path_commments", type=str, help="Path of dataset in S3 for reddit comments")
    parser.add_argument("--s3_dataset_path_submissions", type=str, help="Path of dataset in S3 for reddit submissions")
    parser.add_argument("--s3_output_bucket", type=str, help="s3 output bucket")
    parser.add_argument("--s3_output_prefix", type=str, help="s3 output prefix")
    parser.add_argument("--subreddits", type=str, help="comma separate list of subreddits of interest")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()
    logger.info(f"spark version = {spark.version}")
    
    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    sc = spark.sparkContext
    sc._jsc.hadoopConfiguration().set(
        "mapred.output.committer.class", "org.apache.hadoop.mapred.FileOutputCommitter"
    )

   
    # Downloading the data from S3 into a Dataframe
    logger.info(f"going to read {args.s3_dataset_path_commments}")
    comments = spark.read.parquet(args.s3_dataset_path_commments, header=True)
    logger.info(f"finished reading files...")
    
    logger.info(f"going to read {args.s3_dataset_path_submissions}")
    submissions = spark.read.parquet(args.s3_dataset_path_submissions, header=True)
    logger.info(f"finished reading files...")
    
    # filter the dataframe to only keep the subreddits of interest
    subreddits = [s.strip() for s in args.subreddits.split(",")]
    excluded_cols = [] # ['edited', 'created_utc', 'retrieved_on']
    
    comments_included_cols = [c for c in comments.columns if c not in excluded_cols]
    logger.info(f"comments included_cols={comments_included_cols}")

    submissions_included_cols = [c for c in submissions.columns if c not in excluded_cols]
    logger.info(f"submissions included_cols={submissions_included_cols}")

    # subset the dataframes because "edited" and "created_utc" have data type problems
    # sometimes they occur as int some time as float and since schema is encoded in the 
    # parquet files therefore different files have different data tpyes for these fields (float in some cases, int in some cases)
    # and spark enforces strict type checking on read so the only option we have is to either
    # fix this outside of spark or delete these columns.
    comments = comments.select(comments_included_cols)
    submissions = submissions.select(submissions_included_cols)
    
    submissions_filtered = submissions.where(lower(col("subreddit")).isin(subreddits))
    comments_filtered = comments.where(lower(col("subreddit")).isin(subreddits))
    
    # save the filtered dataframes so that these files can now be used for future analysis
    s3_path = f"s3://{args.s3_output_bucket}/{args.s3_output_prefix}/comments_3"
    logger.info(f"going to write comments for {subreddits} in {s3_path}")
    logger.info(f"shape of the comments_filtered dataframe is {comments_filtered.count():,}x{len(comments_filtered.columns)}")
    comments_filtered.write.mode("overwrite").parquet(s3_path)
    
    s3_path = f"s3://{args.s3_output_bucket}/{args.s3_output_prefix}/submissions_3"
    logger.info(f"going to write submissions for {subreddits} in {s3_path}")
    logger.info(f"shape of the submissions_filtered dataframe is {submissions_filtered.count():,}x{len(submissions_filtered.columns)}")
    submissions_filtered.write.mode("overwrite").parquet(s3_path)

if __name__ == "__main__":
    main()
