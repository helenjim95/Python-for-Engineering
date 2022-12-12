import joblib
import pandas as pd
from matplotlib import pyplot, pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

def main():
    dataframe = pd.read_csv("__files/credit.csv", header=0, index_col=0)
    # print(dataframe.shape) # (400 samples, 11 columns)
    print(dataframe.head())
    data = dataframe.values
    dataframe_x = dataframe[["Income", "Limit", "Rating", "Cards"]]
    X = dataframe_x.values
    y = data[:, -1]
    n_sample = X.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8)

    alpha_range = [float(x) for x in range(1, 10001)]
    columns = ['Alpha', 'Income', 'Limit', 'Rating', 'Cards', 'R2_score', 'MAE']
    df = pd.DataFrame(columns=columns)

    for alpha in alpha_range:
        print(f"Alpha: {alpha}")
        index = alpha_range.index(alpha)
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        coefficients = model.coef_
        r2 = r2_score(y_test, y_pred)
        df.loc[index, ['Alpha']] = alpha
        df.loc[index, ['Income']] = model.coef_[0]
        df.loc[index, ['Limit']] = model.coef_[1]
        df.loc[index, ['Rating']] = model.coef_[2]
        df.loc[index, ['Cards']] = model.coef_[3]
        df.loc[index, ['R2_score']] = r2_score(y_test, y_pred)
        df.loc[index, ['MAE']] = mean_absolute_error(y_test, y_pred)
        if index < 10:
            s = joblib.dump(model, f"model_{index + 1}.joblib")

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    for (columnName, columnData) in df.iteritems():
        if columnName != 'Alpha' and columnName != 'R2_score' and columnName != 'MAE':
            ax1.plot(df['Alpha'], columnData, label=columnName)
    ax1.set(ylabel='Value of the coefficient', ylim=(-10, 10))
    ax1.legend(loc="upper right")
    ax2.plot(df['Alpha'], df['R2_score'], label='R2_score')
    ax2.set(ylabel='R square', ylim=(0.7, 1))
    plt.xscale('log')
    plt.xlabel('alpha')
    plt.savefig("plot.pdf")
    plt.show()



if __name__ == "__main__":
    main()
