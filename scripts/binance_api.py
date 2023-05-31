import json

from binance import ThreadedWebsocketManager, Client
from sqlalchemy import create_engine, Table, Column, String, MetaData

# create the engine to connect to the database
engine = create_engine(
    "postgresql://sm_user:4f[&<I(76KJz@smarthomedb.cc4jofnmxtin.eu-central-1.rds.amazonaws.com:5432/smart_home")

# create metadata object to define tables
metadata = MetaData(schema='hft')

# define the table with text field data
lob_table = Table('lob_data', metadata,
                  Column('data', String)
                  )

# define another table with text field data
kline_table = Table('kline_data', metadata,
                    Column('data', String)
                    )

api_key = "egaCKG9NhwPkVP8cQwneP5wmrWBdA0jH2bt74lzYIwejRvlBRjedVrv25PZdKQLd"
api_secret = "09YaTI319vmM0u2h4ySyD1HTwujyigtkqGfFJcynCp0CxMpmaexCM7S22T5dtQaD"


def main():
    conn = engine.connect()
    symbol = 'BTCUSDT'
    client = Client(api_key=api_key, api_secret=api_secret)
    depth = client.get_order_book(symbol=symbol)
    conn.execute(lob_table.insert().values(data=json.dumps(depth)))
    print(depth)


    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
    # start is required to initialise its internal loop
    twm.start()

    def handle_socket_message(msg):
        conn.execute(lob_table.insert().values(data=json.dumps(msg)))
        print(msg)


    streams = ['btcusdt@depth', 'btcusdt@kline_1s']
    twm.start_multiplex_socket(callback=handle_socket_message, streams=streams)

    twm.join()
    conn.close()


if __name__ == "__main__":
    main()