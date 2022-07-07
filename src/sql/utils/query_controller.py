import psycopg2 as pg

def get_query(q, cur):
    cur.execute(q)
    return cur.fetchall(), [row[0] for row in cur.description]

def query_controller(query):
    try:
        DBNAME = "credit_card_classification"
        USER_NAME = "postgres"
        USER_PASWD = "9182"
        HOST = "localhost"
        conn = pg.connect(
            f" \
            dbname={DBNAME} \
            user={USER_NAME} \
            password={USER_PASWD} \
            host={HOST}"
        )
        _cur = conn.cursor()
        return get_query(query, _cur)
    except:
        raise
    finally:
        _cur.close()
        conn.close()