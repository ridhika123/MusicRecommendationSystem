#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:59:12 2023

@author: hoahduong & ridhikaagrawal

Usage:
    $ spark-submit --deploy-mode client LightFM_file_conversion.py 
"""

# Import command line arguments and helper functions(if necessary)
import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, count, rand, sum, collect_list, array, sort_array, slice, struct, lit, explode

    
def main(spark, userID):
    file_names = ['utility_train', 'utility_val']
    file_paths = [f'{x}.parquet' for x in file_names]

    utility_train = spark.read.parquet(file_paths[0])
    utility_val = spark.read.parquet(file_paths[1])

    utility_train.createOrReplaceTempView('utility_train')
    utility_val.createOrReplaceTempView('utility_val')
    print('FINISHED READING DATA')
    
    query_train = "SELECT COUNT(*) FROM utility_train"
    query_val = "SELECT COUNT(*) FROM utility_val"
    
    count_train = spark.sql(query_train)
    print('UTILITY_TRAIN BEFORE DROPPING')
    count_train.show()
    
    count_val = spark.sql(query_val)
    print('UTILITY_VAL BEFORE DROPPING')
    count_val.show()
    
    grouped_by_users = utility_train.groupby('user_id').count()\
                                    .withColumnRenamed('count', 'song_count_per_user')            
    grouped_by_items = utility_train.groupby('song_id').count()\
                                    .withColumnRenamed('count', 'user_count_per_song')
    
    grouped_by_users.createOrReplaceTempView('grouped_by_users')
    grouped_by_items.createOrReplaceTempView('grouped_by_items')
                                           
    query_songs = "SELECT COUNT(*) FROM grouped_by_items"
    query_user = "SELECT COUNT(*) FROM grouped_by_users"
    
    count_song = spark.sql(query_songs)
    print('NUMBER OF SONGS BEFORE DROPPING')
    count_song.show()
    
    count_user = spark.sql(query_user)
    print('NUMBER OF USERS')
    count_user.show()
    
    songs_filtered3 = grouped_by_items.filter(grouped_by_items.user_count_per_song > 3)
    # print('SONGS CREATED')
    
    songs_filtered3.createOrReplaceTempView('songs_filtered3')
                                           
    query_songs = "SELECT COUNT(*) FROM songs_filtered3"
    
    count_song = spark.sql(query_songs)
    print('NUMBER OF SONGS AFTER DROPPING')
    count_song.show()
    
    # utility_train_filtered = songs_filtered3.join(utility_train, 'song_id', 'right')\
    #                                         .dropna(subset = ['user_count_per_song'])\
    #                                         .select('user_id', 'song_id', 'total_count')
    # utility_val_filtered = songs_filtered3.join(utility_val, 'song_id', 'right')\
    #                                         .dropna(subset = ['user_count_per_song'])\
    #                                         .select('user_id', 'song_id', 'total_count')
    
    # utility_train_filtered = songs_filtered3.join(utility_train, 'song_id', 'inner').select('user_id', 'song_id', 'total_count')
    # utility_val_filtered = songs_filtered3.join(utility_val, 'song_id', 'inner').select('user_id', 'song_id', 'total_count')
          
    windowSpec = Window.partitionBy("song_id").orderBy("song_id")
    df_with_count_train = utility_train.select("*", count("*").over(windowSpec).alias("user_count"))
    df_with_count_val = utility_val.select("*", count("*").over(windowSpec).alias("user_count"))
    
    # Filter songs with more than three listeners
    utility_train_filtered = df_with_count_train.filter(df_with_count_train.user_count > 3).drop("user_count")
    utility_val_filtered = df_with_count_val.filter(df_with_count_val.user_count > 3).drop("user_count")
                                  
    utility_train_filtered.createOrReplaceTempView('utility_train_filtered')
    utility_val_filtered.createOrReplaceTempView('utility_val_filtered')                                                                               
    
    query_train = "SELECT COUNT(*) FROM utility_train_filtered"
    query_val = "SELECT COUNT(*) FROM utility_val_filtered"
    
    count_train = spark.sql(query_train)
    print('UTILITY_TRAIN AFTER DROPPING')
    count_train.show()
    
    count_val = spark.sql(query_val)
    print('UTILITY_VAL AFTER DROPPING')
    count_val.show()
    
    # utility_train_filtered.write.mode('overwrite').parquet('utility_train_filtered.parquet')
    # print('WRITTEN TRAIN')
    # utility_val_filtered.write.mode('overwrite').parquet('utility_val_filtered.parquet')
    # print('WRITTEN VAL')
    
    utility_train_filtered_rep = utility_train_filtered.repartition('user_id')
    utility_val_filtered_rep = utility_val_filtered.repartition('user_id')
    
    utility_train_filtered_rep.write.mode('overwrite').parquet('utility_train_filtered_rep.parquet')
    print('WRITTEN TRAIN')
    utility_val_filtered_rep.write.mode('overwrite').parquet('utility_val_filtered_rep.parquet')
    print('WRITTEN VAL')

    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('data_preprocessing').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']
    
    # Get file_path for dataset to analyze
    # file_path = sys.argv[1]

    # Call our train routine
    main(spark, userID)

