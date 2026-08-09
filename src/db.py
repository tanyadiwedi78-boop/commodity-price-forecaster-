import sys , os

sys.path.append(".")
sys.path.append("..")

from sqlalchemy import create_engine , text
from urllib.parse import quote_plus
from config.settings import DB_HOST , DB_PORT , DB_NAME , DB_USER , DB_PASSWORD


def get_engine():
    """ create a SQLALchemy engine connected to PostgreSQL"""
    safe_password = quote_plus(DB_PASSWORD)
    safe_user = quote_plus(DB_USER)
    url = f"postgresql+psycopg2://{safe_user}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)

def init_tables():
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS  forecasts (
                id                       SERIAL PRIMARY KEY,
                commodity                VARCHAR(50) NOT NULL,
                forecast_date            DATE NOT NULL,
                predicted_price          NUMERIC(14 , 2) NOT NULL,
                predicted_price_lgbm     NUMERIC(14 , 2),
                predicted_price_sarima   NUMERIC(14 , 2),
                ci_low                  NUMERIC(14,2),
                ci_high                 NUMERIC(14,2),
                horizon_days            INTEGER NOT NULL,
                model_version            VARCHAR(30),
                recommendation          VARCHAR(10),
                reasoning               TEXT,
                pct_change              NUMERIC(8,2),
                generated_at             TIMESTAMP DEFAULT NOW()
                
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_metrics(
                id              SERIAL PRIMARY KEY,
                commodity       VARCHAR(50) NOT NULL,
                model_name     VARCHAR(20) NOT NULL,
                rmse           NUMERIC(14 ,4),
                mape           NUMERIC(8 , 2),
                trained_at     TIMESTAMP DEFAULT NOW());
            """))

        # useful indexes for the queries your FASTAPI backend will run
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_forecasts_commodity_date ON forecasts (commodity , forecast_date);
            """))

        conn.commit()

    print("Tables ready: forecasts , model_metrics")

if __name__ == "__main__":
    init_tables()