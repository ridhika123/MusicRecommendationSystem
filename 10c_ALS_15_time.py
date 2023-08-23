#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:59:12 2023

@author: hoahduong & ridhikaagrawal

Usage:
    $ spark-submit --deploy-mode client ALS_full.py 
"""

# Import command line arguments and helper functions(if necessary)
import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import row_number, count, rand, sum, collect_list, array, sort_array, slice, struct, lit, explode
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.feature import StringIndexer, IndexToString
import time
"""
/user/bm106_nyu_edu/1004-project-2023/

interactions_test.parquet   
interactions_train_small.parquet  
interactions_train.parquet
"""
   
def train_and_eval(spark, userID, file_paths, r, lambdaReg, a, utility_train, utility_val, selected_val):
    
    start_time = time.time()
    print("START TIME", start_time)
    
    print('STARTING FITTING MODELS')

    als = ALS(maxIter = 10, regParam = lambdaReg, rank = r, alpha = a, userCol = 'user_id', itemCol = 'song_id', \
              ratingCol = 'total_count', coldStartStrategy = 'drop', implicitPrefs = True)
    
    model = als.fit(utility_train)
    
    print('FINISHED FITTING MODELS')

    end_time = time.time()
    print("END TIME", end_time)
    print("TIME TO TRAIN", end_time-start_time)
    
    # users_val = utility_val.select('user_id').distinct()
    
    # # train_recs = model.recommendForAllUsers(100)
    # val_recs = model.recommendForUserSubset(users_val, 100)
    
    # print('FINISHED RECOMMENDATIONS')
    
    # final_val_recs = val_recs.select('user_id', val_recs.recommendations.song_id)\
    #                   .withColumnRenamed('recommendations.song_id', 'song_id')

    # print('FINISHED FINAL RECOMMENDATIONS')
    
    # pred_and_actual = final_val_recs.join(selected_val, final_val_recs.user_id == selected_val.user_id, 'left')\
    #                             .select('song_id', 'actual_songs')
                                
    # print('FINISHED PRED_AND_ACTUAL')
                                
    # metrics = RankingMetrics(pred_and_actual.rdd)
    
    # print('FINISHED CALCULATING METRICS')

    # map_at_100 = metrics.meanAveragePrecisionAt(100)
    # ncdg_at_100 = metrics.ndcgAt(100)
    # precision_at_100 = metrics.precisionAt(100)
    
    # return map_at_100, ncdg_at_100, precision_at_100


def train_and_test(spark, userID, file_paths, r, lambdaReg, a):
    
    utility_train = spark.read.parquet(file_paths[0])
    utility_test = spark.read.parquet(file_paths[3])

    utility_train.createOrReplaceTempView('utility_train')
    utility_test.createOrReplaceTempView('utility_test')

    print('FINISHED READING DATA')

    als = ALS(maxIter = 10, regParam = lambdaReg, rank = r, alpha = a, userCol = 'user_id', itemCol = 'song_id', \
              ratingCol = 'total_count', coldStartStrategy = 'drop', implicitPrefs = True)
    
    model = als.fit(utility_train)
    
    print('FINISHED FITTING MODELS')
    
    users_test = utility_test.select('user_id').distinct()
    
    # train_recs = model.recommendForAllUsers(100)
    test_recs = model.recommendForUserSubset(users_test, 100)
    
    print('FINISHED RECOMMENDATIONS')
    
    final_test_recs = test_recs.select('user_id', test_recs.recommendations.song_id)\
                      .withColumnRenamed('recommendations.song_id', 'song_id')

    print('FINISHED FINAL RECOMMENDATIONS')
    
    # Actual songs
    
    grouped_test = utility_test.groupBy('user_id')\
                            .agg(slice(sort_array(collect_list(struct('total_count', 'song_id')), asc = False), 1, 100).alias('songs'))
    selected_test = grouped_test.select('user_id', grouped_test.songs.song_id)\
                        .withColumnRenamed('songs.song_id', 'actual_songs')
                        
    print('FINISHED ACTUAL SONGS')
    
    pred_and_actual = final_test_recs.join(selected_test, final_test_recs.user_id == selected_test.user_id, 'left')\
                                .select('song_id', 'actual_songs')
                                
    print('FINISHED PRED_AND_ACTUAL')
                                
    metrics = RankingMetrics(pred_and_actual.rdd)
    
    print('FINISHED CALCULATING METRICS')

    map_at_100 = metrics.meanAveragePrecisionAt(100)
    return map_at_100
    
    
def main(spark, userID):
    # file_names = ['interactions_train_small', 'interactions_train', 'interactions_val_small', 'interactions_val']
    
    file_names = ['utility_train_filtered15', 'utility_val_filtered15', 'selected_val', 'utility_test']
    file_paths = [f'{x}.parquet' for x in file_names]
    # file_paths.append('utility_test.parquet')
    
    utility_train = spark.read.parquet(file_paths[0])
    utility_val = spark.read.parquet(file_paths[1])

    utility_train.createOrReplaceTempView('utility_train')
    utility_val.createOrReplaceTempView('utility_val')
    
    # utility_train.show() ####################
    # utility_val.show()

    selected_val = spark.read.parquet(file_paths[2])
    selected_val.createOrReplaceTempView('selected_val')
    # selected_val.show() ####################
    
    maps_val = []
    ndcg_val = []
    precision_val = []
    
    # ranks = [10]
    alphas = [1.0]
    # regParams = [0.1]
    
    ranks = [40]
    # alphas = [0.1, 10]
    regParams = [0.2]
    
    for r in ranks:
        for a in alphas:
            for lambdaReg in regParams:
                train_and_eval(spark, userID, file_paths, r, lambdaReg, a, utility_train, utility_val, selected_val)
                # print('MeanAP on validation set for (rank, lambda, alpha) =', r, lambdaReg, a, ':', curr_map_val)
                # print('NCDG on validation set for (rank, lambda, alpha) =', r, lambdaReg, a, ':', curr_ndcg_val)
                # print('Precision on validation set for (rank, lambda, alpha) =', r, lambdaReg, a, ':', curr_precision_val)
                # maps_val.append(curr_map_val)
                
    # best_rank = 10
    # best_alpha = 1.0
    # best_regParams = 0.1
    
    # map_test = train_and_test(spark, userID, file_paths, best_rank, best_regParams, best_alpha)
    # print('MeanAP on test set for (rank, lambda, alpha) =', best_rank, best_regParams, best_alpha, ':', map_test)
    
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('ALS_full_10').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']
    
    # Get file_path for dataset to analyze
    # file_path = sys.argv[1]

    # Call our train routine
    main(spark, userID)

