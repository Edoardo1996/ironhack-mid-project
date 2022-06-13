CREATE DATABASE IF NOT EXISTS credit_card_classification;

CREATE TYPE offer_accepted_type AS ENUM ('Yes', 'No');
CREATE TYPE reward_type AS ENUM ('Air Miles', 'Cash Back', 'Points');
CREATE TYPE mailer_type_type AS ENUM ('Letter', 'Postcard');
CREATE TYPE income_level_type AS ENUM ('High', 'Medium', 'Low');
CREATE TYPE overdraft_protection_type AS ENUM ('Yes', 'No');
CREATE TYPE credit_rating_type AS ENUM ('High', 'Medium', 'Low');
CREATE TYPE own_your_home_type AS ENUM ('Yes', 'No');

CREATE TABLE IF NOT EXISTS credit_card_data (
	customer_number serial PRIMARY KEY,
	offer_accepted offer_accepted_type NOT NULL,
	reward reward_type NOT NULL,
    mailer_type mailer_type_type NOT NULL,
    income_level income_level_type NOT NULL, 
    bank_accounts_open INT NOT NULL,
    overdraft_protection overdraft_protection_type NOT NULL,
    credit_rating credit_rating_type NOT NULL,
    credit_cards_held INT NOT NULL,
    homes_owned INT NOT NULL,
    household_size INT NOT NULL,
    own_your_home own_your_home_type NOT NULL,
    average_balance NUMERIC (18, 2), 
    q1_balance INT,
    q2_balance INT,
    q3_balance INT,
    q4_balance INT
);
