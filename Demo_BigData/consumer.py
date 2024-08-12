import json 
from kafka import KafkaConsumer

if __name__ == '__main__':
    # Kafka Consumer 
    consumer = KafkaConsumer(
        'messages',
        bootstrap_servers='localhost:9093',
        auto_offset_reset='earliest'
    )
    for message in consumer:
        try:
            my_bytes_value = message.value
            my_json = my_bytes_value.decode('utf8').replace("'", '"')
            print(json.loads(my_json))
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
