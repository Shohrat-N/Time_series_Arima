import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_predict

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.arima.model import ARIMAResults
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)


import scipy
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf


import streamlit as st

header=st.container()
dataset=st.container()
features=st.container()
model=st.container()
forecastt=st.container()


with header:
    st.title("Time Series: Population of Azerbaijan")

with header:
    st.header("Through the Arima model, the number of people in Azerbaijan is forecasted for the next 7 years.")



data=pd.read_excel("/content/Population_76.xlsx")
data.head()

data.Year=data.Year.astype(int).astype(str) + '-01-01'
data.Year=pd.to_datetime(data.Year)

data.sort_values(by='Year', ascending=True, inplace=True)
df=data.copy()

df.reset_index(drop=True, inplace=True)
df["Year"] =  pd.to_datetime(df['Year'], format='%Y.%M.%d').dt.strftime('%Y/%m/%d')

data.head()
data.describe()

with dataset:
    st.subheader("Population of Azerbaijan Dataset")
    st.write(df.head(73))
    
with dataset:
    st.subheader("Statistical Properties")
    st.write(data.describe().transpose())
 
data.set_index('Year', drop=True, inplace=True)

fig, ax=plt.subplots()
sns.set_style('darkgrid')
data.Population.plot(color='green', figsize=(16, 8), lw=3)
plt.title('Azerbaijani Population over Years')

with dataset:
    st.subheader("Visualisation of Population through Years")
    st.pyplot(fig)
            


fig, ax=plt.subplots()
decomposition=seasonal_decompose(data['Population'])
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)

with dataset:
    st.subheader("Decomposition")
    st.pyplot(fig)
    st.write("From the graph we can see that the entire series is taken as the trend component and there is no seasonality. We can also see that the residual plot shows around zero.")

result_adfuller = adfuller(data['Population'])
result_adfuller=pd.DataFrame(result_adfuller)
result_adfuller.drop(labels=[2, 3, 4, 5], axis=0,  inplace=True)
result_adfuller.columns=["ADF test result"]
result_adfuller.index=["ADF test statistic", "p value"]

result_kpss = kpss(data.Population)
result_kpss=pd.DataFrame(result_kpss)
result_kpss.drop(labels=[2, 3], axis=0,  inplace=True)
result_kpss.columns=["KPSS test result"]
result_kpss.index=["KPSS test statistic", "p value"]

result_adfuller_1 = adfuller(data['Population'])
result_adfuller_1=pd.DataFrame(result_adfuller_1)
result_adfuller_1.drop(labels=[2, 3, 4, 5], axis=0,  inplace=True)
result_adfuller_1.columns=["ADF test result"]
result_adfuller_1.index=["ADF test statistic", "p value"]


result_kpss_1 = kpss(data.Population)
result_kpss_1=pd.DataFrame(result_kpss_1)
result_kpss_1.drop(labels=[2, 3], axis=0,  inplace=True)
result_kpss_1.columns=["KPSS test result"]
result_kpss_1.index=["KPSS test statistic", "p value"]

with dataset:
    st.subheader("Testing for Stationarity")
    
with dataset:   
    col1, col2 = st.columns(2)
    col1.write(result_adfuller)
    col2.write(result_kpss)
with dataset:
    st.write("According to the results of Adfuller and KPSS tests, given the p value that is more than 0.05 in Adfuller and less 0.05 in KPSS, we fail to reject Null Hypothesis that is data is non-stationary. It needs transformation. After applying first-order differencing the results of Adfuller and KPSS tests as following:")

with dataset:   
    coll1, coll2 = st.columns(2)
    coll1.write(result_adfuller_1)
    coll2.write(result_kpss_1)
    st.write("Thus, first-order differencing is enough to make dataset stationary given the result of the test. While building ARIMA model I can do it with q=1")
    

with dataset:
    st.subheader("Autocorrelation")
    acf, pacf=st.columns(2)
    acf.pyplot(plot_acf(data['Population'].diff().dropna()))
    pacf.pyplot(plot_pacf(data['Population'].diff().dropna(), lags=15))
    st.write("Through autocarrelation and partial autocorrelation plots main priority here is to try to find out how many lags I should use for AR or MA components in the ARIMA model. While building the model I will use AR(2).'If the autocorrelation plot shows positive autocorrelation at the first lag (lag-1), then it suggests to use the AR terms in relation to the lag If the autocorrelation plot shows negative autocorrelation at the first lag, then it suggests using MA terms.'")

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf



#order= auto_arima(data, suppress_warnings=True)           
#order.summary()
#order_df=pd.DataFrame(order.summary())
dict_={"Model": "SARIMAX(0, 1, 0)", "No.Observations": "73", "Sample": "01/01/1950, 01/01/2022", "AIC": "1667.315", "P>|z|": "intercept -0.000", "Ljung-Box":" Prob(Q-	0.00	Prob(JB)-0.15" }
dict_df=pd.DataFrame(dict_, index=["Result"])

with model:
    st.subheader("Auto Arima model")
    st.dataframe(dict_df)

from PIL import Image
image = Image.open('/content/Untitled.png')

with model:
    st.subheader("Error Diagnostics for Auto Arima")
    st.image(image)
    st.write("According to the graph and Ljung-Box test, we reject to Null Hypothesis that residuals are not uncorrelated. That's why I'll not be able to use this model.")


train=data[:60]
test=data[60:]

fig, ax=plt.subplots(figsize=(16, 8))
sns.set_style('darkgrid')
ax.plot(train, label='Train', lw=3)
ax.plot(test, label='Test', lw=3)
ax.legend()
with model:
    st.subheader("Train-Test Split")
    st.pyplot(fig)

model_arima=ARIMA(train, order=(2, 1, 0))
model_arima_fit=model_arima.fit()

with model:
    st.subheader("Building Arima Model on Train Dataset")
    st.write(model_arima_fit.summary())

residuals = model_arima_fit.resid
box_test=acorr_ljungbox(residuals, np.arange(1, 11, 1), return_df=True)

fig, ax=plt.subplots()
model_arima_fit.resid.plot(kind='kde')
with model:
    st.subheader("Error Diagnostics for Arima Model")
    st.write("I will also run the model diagnostics check for the assumptions Normality of errors and the distribution of residuals. So the errors are should be normally distributed and uncorrelated to each other.")
    m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #00ff00;
    color:#ff0000;
    }
</style>""", unsafe_allow_html=True)
    if st.button('Lyung Box Test', disabled=False):
       st.write(box_test)
    if st.button('Normality of Residuals', disabled=False):
       st.pyplot(fig)
    st.write("As p-value is more than 0.05 according to the result of Lyung Box test, we fail to reject Null Hyphotesis that residuals are uncorrolated")



fig, ax=plt.subplots()
plt.show()
with model:
    st.subheader("Result of Train dataset")
    st.pyplot(model_arima_fit.plot_predict(dynamic=False))

start=len(train)
end=len(train)+len(test)-1
pred=model_arima_fit.predict(start=start,end=end,typ='levels').rename('ARIMA predictions')
data["Prediction"]=model_arima_fit.predict(start=start,end=end,typ='levels')

without_t=data.copy()
without_t.index=without_t.index.date

from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
mape=sqrt(mean_absolute_percentage_error(test['Population'], pred))

fig, ax=plt.subplots()
plt.show()
pred.plot(legend=True)
test['Population'].plot(legend=True)
with model:
    st.subheader("Prediction of Test Dataset")
    st.pyplot(fig)
    st.write(without_t.tail(13))
    st.write("Measure of prediction accuracy with mean absolute percentage error")
    st.write(mape)

# fit the model
model2=ARIMA(data['Population'],order=(2,1,0))
model2=model2.fit()

#forecast next 7 years
forecast=model2.predict(start=len(data), end=len(data)+7,typ='levels').rename('Forecasting')
forc=forecast.copy()
forc.index=forc.index.date



fig, ax=plt.subplots()
forecast.plot(legend=True, lw=3)
data['Population'].plot(legend=True, lw=2)
with forecastt:
    st.subheader("Forecast Next 7 Years")
    st.pyplot(fig)
    st.write(forc)

with forecastt:
    st.text("Source of dataset: https://www.macrotrends.net/countries/AZE/azerbaijan/population#:~:text=The%20population%20of%20Azerbaijan%20in,a%200.79%25%20increase%20from%202018")
    st.text("Shohrat Naghiyeva")
