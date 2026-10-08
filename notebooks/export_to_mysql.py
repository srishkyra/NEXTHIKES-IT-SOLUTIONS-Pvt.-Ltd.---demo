import os, pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.types import String

user, pw = os.environ['TELLCO_MYSQL_USER'], os.environ['TELLCO_MYSQL_PASSWORD']
host, db = os.environ.get('TELLCO_MYSQL_HOST', 'localhost'), os.environ.get('TELLCO_MYSQL_DB', 'telecom_db')
engine = create_engine(URL.create('mysql+pymysql', username=user, password=pw, host=host, database=db))

df = pd.read_csv('outputs/customer_satisfaction_scores.csv', dtype={'MSISDN': str})
df.to_sql('customer_satisfaction_scores', engine, if_exists='replace', index=False,
          dtype={'MSISDN': String(20), 'Scoring Version': String(60), 'Scoring Timestamp': String(40)})
with engine.begin() as c:
    c.execute(text('ALTER TABLE customer_satisfaction_scores ADD PRIMARY KEY (MSISDN)'))
    print(c.execute(text('SELECT COUNT(*) FROM customer_satisfaction_scores')).scalar(), 'rows exported')
print(pd.read_sql('SELECT MSISDN, `Satisfaction Score`, `Engagement Score`, `Experience Score` '
                  'FROM customer_satisfaction_scores ORDER BY `Satisfaction Score` DESC LIMIT 10', engine))
