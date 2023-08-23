#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 14:01:37 2023

@author: hoahduong & ridhikaagrawal

Usage:
    $ spark-submit --deploy-mode client indexer_v2.py 
"""

# Import command line arguments and helper functions(if necessary)
import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import monotonically_increasing_id, lit, row_number
from pyspark.sql.window import Window

"""
/user/bm106_nyu_edu/1004-project-2023/

tracks_test.parquet
tracks_train.parquet
tracks_train_small.parquet
"""

def main(spark, userID):
    
    file_names = ['tracks_train_small', 'tracks_train']
    file_paths = [f'hdfs:/user/bm106_nyu_edu/1004-project-2023/{x}.parquet' for x in file_names]
    
    for i in range(len(file_names)):
        name = file_names[i]
        path = file_paths[i]
        
        tracks = spark.read.parquet(path)
        
        # Give the dataframe a temporary view so we can run SQL queries
        tracks.createOrReplaceTempView('tracks')
        
        # Indexer
        
        w = Window().orderBy(lit('A'))
        indexed_tracks = tracks.withColumn("song_id", row_number().over(w))
        print('FINISHED ADDING SONG_ID')
        
        indexed_tracks.write.mode('overwrite').parquet(f'{name}_indexed_row.parquet')
    
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