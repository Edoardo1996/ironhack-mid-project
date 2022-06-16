"""Test classification scoring algorithm"""
from src.utils.cleaning import *
import src.utils.sktools as skt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
pd.set_option('display.max_columns', None)

def main():
    df = load_data(config.cleaned_data_path)
    # Benchmark Logistic Regressor Model
    cols_ord_encod = ['income_level', 'credit_rating']
    cols_onehot_encod = ['reward', 'mailer_type', 'overdraft_protection', 
                        'own_your_home']
    predictions, classification_report = skt.score_classification_model(
        df=df, target='offer_accepted', model=LogisticRegression(solver='lbfgs'),
        cols_to_encode=[cols_ord_encod, cols_onehot_encod],
        scaler=StandardScaler(),
        encoders=[OrdinalEncoder(), OneHotEncoder()],
        outsiders_thresh=None
    )
    print(classification_report)

if __name__ == '__main__':
    main()