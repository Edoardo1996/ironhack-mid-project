from itertools import groupby


TABLE_NAME = "credit_card_data"
# variable for question-05
COLUMN_TO_DROP = "q4_balance"

# question-04
QUERY_00 = f"\
            select\
                *\
            from {TABLE_NAME};"

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
            from {TABLE_NAME};"

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
            from {TABLE_NAME};"
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

# 10.3
QUERY_08 = f"\
            select\
                credit_rating,\
                round(avg(credit_cards_held), 2) as avg_credit_cards_held\
            from {TABLE_NAME}\
            group by credit_rating;"

# 10.4
QUERY_09 = f"\
            select\
                bank_accounts_open,\
                round(avg(credit_cards_held), 2) as avg_credit_cards_held\
            from {TABLE_NAME}\
            group by bank_accounts_open;"

# question-11
QUERY_10 = f"\
            select\
                *\
            from {TABLE_NAME}\
                where credit_rating != 'Low' and\
                    credit_cards_held < 3 and\
                        own_your_home = 'Yes' and\
                            household_size > 2 and\
                                offer_accepted = 'Yes';"

# question-12
QUERY_11 = f"\
            with total_balance_avg as (\
            select\
                round(avg(average_balance), 2) as avg\
            from credit_card_data\
            )\
            select\
                *\
            from credit_card_data\
            where average_balance < (select * from total_balance_avg);"

# question-13
QUERY_12 = f"\
            create view\
                Customers__Balance_View1 as\
            {QUERY_11}"

# question-14
QUERY_13 = f"\
            select\
                count(*) as qty_people_accepted_offer\
            from {TABLE_NAME}\
            where offer_accepted = 'Yes';\
            "
QUERY_14 = f"\
            select\
                count(*) as qty_people_rejected_offer\
            from {TABLE_NAME}\
            where offer_accepted = 'No';\
            "
# question-15
QUERY_15 = f""

# question-16
QUERY_16 = f"\
            select\
                mailer_type,\
                count(customer_number) as qty_customers\
            from {TABLE_NAME}\
            group by mailer_type;"
        
# question-17        
QUERY_17 = f"\
            select\
                *\
            from {TABLE_NAME}\
            where q1_balance is not null\
            order by q1_balance asc\
            limit 1 offset 10"