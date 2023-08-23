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
    
    track_names = ['tracks_train_small', 'tracks_train']
    interaction_names = ['interactions_train_small', 'interactions_train']
    track_paths = [f'{x}_indexed_mono.parquet' for x in track_names]
    interaction_paths = [f'{x}_cleaned.parquet' for x in interaction_names]
    
    # for i in range(len(interaction_paths)):
    #     track_name = track_names[i]
    #     track_path = track_paths[i]
        
    #     interaction_name = interaction_names[i]
    #     interaction_path = interaction_paths[i]
        
    #     tracks = spark.read.parquet(track_path)
    #     interactions = spark.read.parquet(interaction_path)
        
    #     # Give the dataframe a temporary view so we can run SQL queries
    #     tracks.createOrReplaceTempView('tracks')
    #     interactions.createOrReplaceTempView('interactions')
        
    #     # Join
    #     indexed_int = interactions.join(tracks, 'recording_msid', 'left')
    #     print('FINISHED JOINING MONO FOR', track_name)
        
    #     indexed_int.write.mode('overwrite').parquet(f'{interaction_name}_indexed_mono.parquet')
    #     print('FINISHED WRITING MONO FOR', track_name)
        
        
    track_paths_row = [f'{x}_indexed_row.parquet' for x in track_names]
    
    for i in range(len(interaction_paths)):
        track_name = track_names[i]
        track_path = track_paths_row[i]
        
        interaction_name = interaction_names[i]
        interaction_path = interaction_paths[i]
        
        tracks = spark.read.parquet(track_path)
        interactions = spark.read.parquet(interaction_path)
        
        # Give the dataframe a temporary view so we can run SQL queries
        tracks.createOrReplaceTempView('tracks')
        interactions.createOrReplaceTempView('interactions')
        
        # Join
        indexed_int = interactions.join(tracks, 'recording_msid', 'left')
        print('FINISHED JOINING ROW FOR', track_name)
        
        indexed_int.write.mode('overwrite').parquet(f'{interaction_name}_indexed_row.parquet')
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