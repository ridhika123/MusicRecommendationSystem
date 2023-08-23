#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 18:19:34 2023

@author: hoahduong

Usage:
    $ spark-submit --deploy-mode client baseline_full.py 
"""

# Import command line arguments and helper functions(if necessary)
import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, sum, collect_list, array, sort_array, slice, struct, lit
from pyspark.mllib.evaluation import RankingMetrics

"""
/user/bm106_nyu_edu/1004-project-2023/

interactions_test.parquet   
interactions_train_small.parquet  
interactions_train.parquet
"""

   
def baseline_popularity_1(spark, userID, path, beta):
    interactions = spark.read.parquet(path)
    
    # Give the dataframe a temporary view so we can run SQL queries
    interactions.createOrReplaceTempView('interactions')
    
    utility_matrix = interactions.groupby('user_id', 'song_id').count()\
                                .withColumnRenamed('count', 'total_count')  
    
    popularity_matrix = utility_matrix.groupby('song_id')\
                                    .agg((sum('total_count') / (count('total_count') + beta)).alias('popularity'))

    popularity_matrix = popularity_matrix.orderBy(popularity_matrix.popularity.desc()).limit(100)
    
    return popularity_matrix

def baseline_popularity_2(spark, userID, path):
    interactions = spark.read.parquet(path)
    
    # Give the dataframe a temporary view so we can run SQL queries
    interactions.createOrReplaceTempView('interactions')
    
    utility_matrix = interactions.groupby('user_id', 'song_id').count()\
                .withColumnRenamed('count', 'total_count')  
    
    popularity_matrix = utility_matrix.groupby('song_id')\
                            .agg((sum('total_count') * (count('total_count'))).alias('popularity'))

    popularity_matrix = popularity_matrix.orderBy(popularity_matrix.popularity.desc()).limit(100)
    
    return popularity_matrix
    
   
def evaluate(spark, userID, val_path, popularity_matrix):
    interactions_val = spark.read.parquet(val_path)
    interactions_val.createOrReplaceTempView('interactions_val')
    
    filled_val_utility = interactions_val.groupBy('user_id', 'song_id').count()\
                                        .withColumnRenamed('count', 'total_count') 
    
    grouped_val = filled_val_utility.groupBy('user_id')\
                .agg(slice(sort_array(collect_list(struct('total_count', 'song_id')), asc = False), 1, 100).alias('songs'))
    
    # print('grouped_val finished!!!!!!!!!')
                    
    popular_songs = popularity_matrix.agg(collect_list('song_id')).collect()[0][0]
    
    # print('popular_songs finished!!!!!!!!!')
    
    selected_val = grouped_val.select(grouped_val.songs.song_id)\
                        .withColumnRenamed('songs.song_id', 'actual_songs')\
                        .withColumn('popular_songs', array([lit(i) for i in popular_songs]))
                
    pred_and_actual_rdd = selected_val.select('popular_songs', 'actual_songs').rdd
    
    metrics = RankingMetrics(pred_and_actual_rdd)
    # print('metrics finished!!!!!!!!!')
    
    map_at_100 = metrics.meanAveragePrecisionAt(100)
    ndcg_at_100 = metrics.ndcgAt(100)
    
    return map_at_100, ndcg_at_100

def evaluate_on_test(spark, userID, test_path, popularity_matrix):
    interactions_test = spark.read.parquet(test_path)
    interactions_test.createOrReplaceTempView('interactions_test')
    
    filled_test_utility = interactions_test.groupBy('user_id', 'song_id').count()\
                                        .withColumnRenamed('count', 'total_count') 
    
    grouped_test = filled_test_utility.groupBy('user_id')\
                .agg(slice(sort_array(collect_list(struct('total_count', 'song_id')), asc = False), 1, 100).alias('songs'))
    
    # print('grouped_val finished!!!!!!!!!')
                    
    popular_songs = popularity_matrix.agg(collect_list('song_id')).collect()[0][0]
    
    # print('popular_songs finished!!!!!!!!!')
    
    selected_test = grouped_test.select(grouped_test.songs.song_id)\
                        .withColumnRenamed('songs.song_id', 'actual_songs')\
                        .withColumn('popular_songs', array([lit(i) for i in popular_songs]))
                
    pred_and_actual_rdd = selected_test.select('popular_songs', 'actual_songs').rdd
    
    metrics = RankingMetrics(pred_and_actual_rdd)
    # print('metrics finished!!!!!!!!!')
    
    map_at_100 = metrics.meanAveragePrecisionAt(100)
    ndcg_at_100 = metrics.ndcgAt(100)
    
    return map_at_100, ndcg_at_100
            

def main(spark, userID):
    # file_names = ['interactions_train_small', 'interactions_train', 'interactions_val_small', 'interactions_val']
    
    file_names = ['interactions_train', 'interactions_val']
    file_paths = [f'{x}_split_row.parquet' for x in file_names]
    file_paths.append('interactions_test_indexed_row.parquet')
    
    # betas = [10, 100, 200, 300, 400, 500]
    # betas = [500, 750, 1000, 2000, 3000, 4000, 5000, 10000]
    # betas = [10000, 12000, 14000, 16000, 18000, 20000, 35000, 50000]
    # betas = [50000, 60000, 70000, 80000, 90000, 100000, 120000, 150000]
    
    # print('USING AVERAGE UTILITY AS POPULARITY MEASURE')
    # betas = [10, 100, 1000, 10000, 50000, 100000, 200000, 500000, 800000]
    # maps_val = []
    # maps_train = []
    # ndcg_train = []
    # ndcg_val = []
    
    # for beta in betas:
    #     pop_matrix = baseline_popularity_1(spark, userID, file_paths[0], beta)
    #     # print('finished pop_matrix!!!!!!!!!!!!!!!!!!!!')
        
    #     curr_map_train, curr_ndcg_train = evaluate(spark, userID, file_paths[0], pop_matrix)
    #     print('MeanAP on training set for beta =', beta, ':', curr_map_train)
    #     print('NDCG on training set for beta =', beta, ':', curr_ndcg_train)
    #     maps_train.append(curr_map_train)
    #     ndcg_train.append(curr_ndcg_train)
        
    #     curr_map_val, curr_ndcg_val = evaluate(spark, userID, file_paths[1], pop_matrix)
    #     print('MeanAP on evaluation set for beta =', beta, ':', curr_map_val)
    #     print('NDCG on evaluation set for beta =', beta, ':', curr_ndcg_val)
    #     maps_val.append(curr_map_val)
    #     ndcg_val.append(curr_ndcg_val)
    
    best_beta = 500000
    pop_matrix = baseline_popularity_1(spark, userID, file_paths[0], best_beta)
    map_test, ndcg_test = evaluate(spark, userID, file_paths[2], pop_matrix)
    print('MeanAP on evaluation set for beta =', best_beta, ':', map_test)
    print('NDCG on evaluation set for beta =', best_beta, ':', ndcg_test)
   
    # print('MeanAP @ 100', maps_val)
    
    print('USING WEIGHTED UTILITY AS POPULARITY MEASURE')
    pop_matrix = baseline_popularity_2(spark, userID, file_paths[0])
    
    # curr_map_train, curr_ndcg_train = evaluate(spark, userID, file_paths[0], pop_matrix)
    # print('MeanAP on training set:', curr_map_train)
    # print('NDCG on training set:', curr_ndcg_train)
    
    # curr_map_val, curr_ndcg_val = evaluate(spark, userID, file_paths[1], pop_matrix)
    # print('MeanAP on evaluation set:', curr_map_val)
    # print('NDCG on evaluation set:', curr_ndcg_val)
    
    map_test, ndcg_test = evaluate(spark, userID, file_paths[2], pop_matrix)
    print('MeanAP on evaluation set for beta =', best_beta, ':', map_test)
    print('NDCG on evaluation set for beta =', best_beta, ':', ndcg_test)
    
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline_full').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']
    
    # Get file_path for dataset to analyze
    # file_path = sys.argv[1]

    # Call our train routine
    main(spark, userID)

