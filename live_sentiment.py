import time
import random
import pickle
from collections import namedtuple

import seaborn as sn
import matplotlib.pyplot as plt
from pyspark import SparkContext, streaming
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import desc

# load model
# model = pickle.load(open('twogram_MNB', 'rb'))
class model:
    @staticmethod
    def predict(object):
        return [random.choice(range(2))]


# fields = ('sentiment', 'count')
# Sentiment = namedtuple('Sentiment', fields)

def classify(tweet):
    label = 'positive' if int(model.predict([tweet])[0]) else 'negative'
    return [label]

def write_results(rdd):
    try:
        results = rdd.collect()
        if len(results) > 0:
            print(results)
            with open('live_results.txt', 'w') as f:
                f.write(str(results[0][1]) + ',' + str(results[1][1]))
    except:
        pass
    return rdd


def live_sentiment():
    ''' Generate multiple streams of twitter data using W workers and P
        partitions. Create a CountMinSketch on each of the P partitions.
        '''
    # create local StreamingContext with two working threads and batch
    #  interval of 10 seconds
    sc = SparkContext('local[2]', 'Spark Count-Min Sketch')
    sqlContext = SQLContext(sc)
    sc.setLogLevel('ERROR')
    ssc = streaming.StreamingContext(sc, 1)

    # create data stream connected to hostname:port
    socket_stream = ssc.socketTextStream('127.0.0.1', 5555)
    lines = socket_stream.window(5)

    # classify each tweet
    # result = lines.flatMap(lambda tweet: (tweet, classify(tweet)))
    # result = lines.flatMap(lambda tweet: classify(tweet))
    # result = lines.flatMap(lambda tweet: classify(tweet))\
    #               .map(lambda sentiment_label: (sentiment_label, 1))\
    #               .reduceByKey(lambda a, b: a + b)
    # (lines.flatMap(lambda tweet: classify(tweet))
    #       .map(lambda sentiment_label: (sentiment_label, 1))
    #       .reduceByKey(lambda a, b: a + b)
    #       .map(lambda rdd: write_results(rdd))
    #       .pprint())
    (lines.flatMap(lambda tweet: classify(tweet))
          .map(lambda sentiment_label: (sentiment_label, 1))
          .reduceByKey(lambda a, b: a + b)
          .foreachRDD(lambda rdd: write_results(rdd)))
    # .map(lambda item: Sentiment(item[0], item[1]))
          # .foreachRDD(lambda rdd: rdd.toDF().sort(desc('count'))
          #             .limit(2).registerTempTable('counts')))
    # sqlContext


    # result.pprint()

    # start context
    ssc.start()


    # count = 0
    # while count < 10:
    #     time.sleep(10)
    #     sentiment_counts = sqlContext.sql('Select sentiment, count from counts')
    #     sentiment_df = sentiment_counts.toPandas()
    #     print(sentiment_df.head)
        # sn.plt.figure(figsize=(10, 8))
        # sn.barplot(x="count", y="tag", data=sentiment_df)
        # sn.plt.show()
        # count += 1

    ssc.awaitTermination()

if __name__ == '__main__':
    live_sentiment()


