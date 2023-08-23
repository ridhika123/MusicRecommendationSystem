#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 15:23:50 2023

@author: hoahduong
"""

# Import command line arguments and helper functions(if necessary)
import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import monotonically_increasing_id


"""
/user/bm106_nyu_edu/1004-project-2023/

tracks_test.parquet
tracks_train.parquet
tracks_train_small.parquet
"""

def main(spark, userID):
    
    track_names = ['tracks_train']
    track_paths_row = [f'{x}_indexed_row.parquet' for x in track_names]
    
    interaction_names = ['interactions_test']
    interaction_paths = [f'hdfs:/user/bm106_nyu_edu/1004-project-2023/{x}.parquet' for x in interaction_names]
        
    track_name = track_names[0]
    track_path = track_paths_row[0]
    
    interaction_name = interaction_names[0]
    interaction_path = interaction_paths[0]
    
    tracks = spark.read.parquet(track_path)
    interactions_test = spark.read.parquet(interaction_path)
    
    # Give the dataframe a temporary view so we can run SQL queries
    tracks.createOrReplaceTempView('tracks')
    interactions_test.createOrReplaceTempView('interactions_test')
    
    # Join
    indexed_int_test = interactions_test.join(tracks, 'recording_msid', 'left')
    indexed_int_test.createOrReplaceTempView('indexed_int_test')
    
    query_count_before = 'SELECT COUNT(*) FROM indexed_int_test'
    count_before = spark.sql(query_count_before)

    print("Count before removing NULLs:")
    count_before.show()
    
    indexed_int_test_dropped = indexed_int_test.dropna(subset = ['song_id'])
    indexed_int_test_dropped.createOrReplaceTempView('indexed_int_test_dropped')
    
    query_count_after = 'SELECT COUNT(*) FROM indexed_int_test_dropped'
    count_after = spark.sql(query_count_after)

    print("Count before removing NULLs:")
    count_after.show()
    
    print('FINISHED JOINING ROW FOR', track_name)
    
    indexed_int_test_dropped.write.mode('overwrite').parquet(f'{interaction_name}_indexed_row.parquet')
    print('FINISHED WRITING ROW FOR', track_name)


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('indexer').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']
    
    # Get file_path for dataset to analyze
    # file_path = sys.argv[1]

    # Call our main routine
    main(spark, userID)