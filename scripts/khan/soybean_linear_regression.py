import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.linear_model import Ridge, LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score

data1 = pd.read_csv('ET.OUT', skiprows=12, header=0,
        delim_whitespace=True, error_bad_lines=False)


df1 = data1[['SRAA','TMAXA', 'TMINA', 'ETAA']]
# print(df1.shape[0])
df1 = df1[~df1.isna().any(axis=1)]
# print(df1.shape[0])
#print(df.columns.values)

df1['SRAA'] = pd.to_numeric(df1['SRAA'], errors='coerce')
df1['TMAXA'] = pd.to_numeric(df1['TMAXA'], errors='coerce')
df1['TMINA'] = pd.to_numeric(df1['TMINA'], errors='coerce')
df1['ETAA'] = pd.to_numeric(df1['ETAA'], errors='coerce')

df1_EO = df1[:128]
df2_EO = df1[129:247]
df3_EO = df1[248:379]
df4_EO = df1[380:]
# print(df1_EO)
# print(df1_EO['ETAA'].values)

df_EO = df1_EO.append([df2_EO, df3_EO, df4_EO])
# print(df_EO.shape[0])
# print(df1)

data2 = pd.read_csv('PlantGro.OUT', skiprows=12, header=0,
        delim_whitespace=True, error_bad_lines=False)

df2 = data2[['LAID']]
# print(df2.shape[0])
df2 = df2[~df2.isna().any(axis=1)]
# print(df2.shape[0])
#df['DAS'] = pd.to_numeric(df['DAS'], errors='coerce')
df2['LAID'] = pd.to_numeric(df2['LAID'], errors='coerce')


df1_LAI = df2[:128]
df2_LAI = df2[134:252]
df3_LAI = df2[258:389]
df4_LAI = df2[395:]
# print(df4_LAI)

df_LAI = df1_LAI.append([df2_LAI, df3_LAI, df4_LAI])
# print(df_LAI.shape[0])


df = pd.concat([df_LAI, df_EO], axis=1, sort = False)
df.columns = ['XHLAI', 'SRAD', 'TMAX', 'TMIN', 'EO']
df['row_no'] = list(range(len(df)))
df = df.set_index('row_no')
# df.set_index(range(len(df)))
# print(df)
# print(df.dtypes)

plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


for x in df.columns:

    # plt.figure(figsize = (10,6))
    df.hist(column = x, color = 'k', bins = 50, alpha = 0.5)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.show()


N = df.shape[0]
train_length = int(np.floor(0.8*N))
test_length = N - train_length
# print(N, train_length, test_length, '\n')
X_train = df.loc[:train_length-1, ['XHLAI', 'SRAD', 'TMAX', 'TMIN']]
y_train = df.loc[:train_length-1, 'EO']
X_test = df.loc[train_length:, ['XHLAI', 'SRAD', 'TMAX', 'TMIN']]
y_test = df.loc[train_length:, 'EO']
# print(X_train.shape[0])
# print(X_test.shape[0])



scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


print('Ridge Model')
ridge_reg = RidgeCV()
ridge_reg.fit(X_train, y_train)
print('Best alpha : {0:.4f}'.format(ridge_reg.alpha_))
print('Best score : {0:.4f}'.format(ridge_reg.score(X_test, y_test)))
coef = pd.Series(ridge_reg.coef_, index = X_train.columns)
imp_coef = coef.sort_values()
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind='barh')
plt.title('Feature importance using Ridge Model')
plt.show()

print('Lasso Model')
lasso_reg = LassoCV()
lasso_reg.fit(X_train, y_train)
print('Best alpha : {0:.4f}'.format(lasso_reg.alpha_))
print('Best score : {0:.4f}'.format(lasso_reg.score(X_test, y_test)))
coef = pd.Series(lasso_reg.coef_, index = X_train.columns)
imp_coef = coef.sort_values()
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind='barh')
plt.title('Feature importance using Lasso Model')
plt.show()



for this_deg in [1, 2]:
    
    print('Order of Polynomial {0} \n'.format(this_deg) )
    
    poly = PolynomialFeatures(degree=this_deg)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.fit_transform(X_test_scaled)

    for this_alpha in [0, 1]:
        
        print('Regularization parameter (alpha) {0} \n'.format(this_alpha))

        linreg = Ridge(alpha=this_alpha).fit(X_train_poly, y_train)

        # print("intercept :", linreg.intercept_)
        # print("parameters are :", linreg.coef_)
        print("train accuracy score :", linreg.score(X_train_poly, y_train))
        print("test accuracy score :", linreg.score(X_test_poly, y_test))


        print('\n')

