TABLE_NAME = "credit_card_data"
# variable for question-05
COLUMN_TO_DROP = "q4_balance"

# question-04
QUERY_00 = f"\
            select\
                *\
            from {TABLE_NAME}"

# question-05
QUERY_01 = f"\
            ALTER TABLE {TABLE_NAME}\
            DROP COLUMN IF EXISTS {COLUMN_TO_DROP};\
            select\
                *\
            from {TABLE_NAME}\
            limit 10;"

# question-06
QUERY_02 = f"\
            select \
                count(*) as qty_rows\
            from {TABLE_NAME}"

# question-07
def query_03():
    lst_col_find_unique = [
        "offer_accepted",
        "reward",
        "mailer_type",
        "credit_cards_held",
        "household_size"
    ]
    lst_query_03 = []
    for _c in lst_col_find_unique:
        lst_query_03.append(
            f"\
            select\
                distinct {_c}\
            from {TABLE_NAME}"
        )
    return lst_query_03

# question-08
QUERY_04 = f"\
            select\
                customer_number,\
                average_balance\
            from {TABLE_NAME}\
            where average_balance is not NULL\
            order by average_balance desc\
            limit 10;"

# question-09
QUERY_05 = f"\
            select\
                round(avg(average_balance), 2) as avg\
            from {TABLE_NAME};"

# question-10
# 10.1
QUERY_06 = f"\
            select\
                income_level,\
                round(avg(average_balance), 2) as avg\
            from {TABLE_NAME}\
            group by income_level;"

# 10.2
QUERY_07 = f"\
            select\
                bank_accounts_open,\
                round(avg(average_balance), 2) as avg\
            from {TABLE_NAME}\
            group by bank_accounts_open;"