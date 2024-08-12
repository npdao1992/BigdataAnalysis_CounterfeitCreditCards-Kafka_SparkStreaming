import time 
import json 
import random 
from datetime import datetime
from data_generator import generate_message
from kafka import KafkaProducer
import logging
from json import dumps, loads
import csv
logging.basicConfig(level=logging.INFO)

# Messages will be serialized as JSON 
def serializer(message):
    return json.dumps(message).encode('utf-8')

# Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9093'],
    value_serializer=serializer
)



if __name__ == '__main__':
    # Infinite loop - runs until you kill the program
    while True:
        with open('onlinefraud_test.csv', 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                print('===============================')
                print(row)
                producer.send('messages', row)
                time_to_sleep = random.randint(1, 10)
                time.sleep(time_to_sleep)

        producer.flush()
