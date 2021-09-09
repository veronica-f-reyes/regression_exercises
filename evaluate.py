
#Function that creates a residual plot and returns regression errors
import matplotlib.pyplot as plt
from math import sqrt


#FUNCTION that creates a residual plot
def plot_residuals(x, y, yhat): 
    
    # residual = actual - predicted
    residual = y - yhat

    # residual plots (y vs residual)
    plt.figure(figsize = (11,5))

    plt.subplot(122)
    plt.scatter(x, residual)
    plt.axhline(y = 0, ls = ':')
    plt.xlabel('x')
    plt.ylabel('Residual')
    plt.title('OLS model residuals')


#FUNCTION that returns regression errors including SSE, ESS, TSS, MSE, RMSE
def regression_errors(y, yhat): 

    # first calculate the square of residuals

    # residual = actual - predicted
    residual = y - yhat
    residual_squared = residual**2

    baseline = y.mean()
    baseline_residual = y - baseline
    baseline_residual_squared = baseline_residual**2

    #sum of squared errors (SSE)
    SSE = residual_squared.sum()
    print('SSE =', "{:.1f}".format(SSE))

    #total sum of squares = SSE for baseline
    TSS =   baseline_residual_squared.sum()
    print("TSS = ","{:.1f}".format(TSS))

    #explained sum of squares (ESS)
    ESS = TSS - SSE
    print("ESS = ","{:.1f}".format(ESS))

    #mean squared error (MSE)
    MSE = SSE/len(y)
    print("MSE = ", "{:.1f}".format(MSE))

    #root mean squared error (RMSE)
    RMSE = sqrt(MSE)
    print("RMSE = ", "{:.1f}".format(RMSE))

    return SSE, ESS, TSS, MSE, RMSE


#FUNCTION that computes the SSE, MSE, and RMSE for the baseline model
def baseline_mean_errors(y):

    #first calculate the square of residuals

    #residual = actual - predicted
    baseline = y.mean()
    baseline_residual = y - baseline
    baseline_residual_squared = baseline_residual**2

    #sum of squared errors (SSE)
    SSE_baseline = baseline_residual_squared.sum()
    print('SSE_baseline =', "{:.1f}".format(SSE_baseline))

    #total sum of squares = SSE for baseline
    TSS_baseline =   baseline_residual_squared.sum()
    print("TSS_baseline = ","{:.1f}".format(TSS_baseline))

    #explained sum of squares (ESS)
    ESS_baseline = TSS_baseline - SSE_baseline
    print("ESS_baseline = ","{:.1f}".format(ESS_baseline))

    #mean squared error (MSE)
    MSE_baseline = SSE_baseline/len(y)
    print("MSE_baseline = ", "{:.1f}".format(MSE_baseline))

    #root mean squared error (RMSE)
    RMSE_baseline = sqrt(MSE_baseline)
    print("RMSE_baseline = ", "{:.1f}".format(RMSE_baseline))

    return SSE_baseline, ESS_baseline, TSS_baseline, MSE_baseline, RMSE_baseline


#FUNCTION that returns true if your model performs better than the baseline, otherwise false
def better_than_baseline(y, yhat): 

 # residual = actual - predicted
    residual = y - yhat
    residual_squared = residual**2

    baseline = y.mean()
    baseline_residual = y - baseline
    baseline_residual_squared = baseline_residual**2

    #sum of squared errors (SSE)
    SSE = residual_squared.sum()
    print('SSE =', "{:.1f}".format(SSE))
    
    #sum of squared errors for baseline (SSE_baseline)
    SSE_baseline = baseline_residual_squared.sum()
    print('SSE_baseline =', "{:.1f}".format(SSE_baseline))

    if SSE_baseline - SSE > 0:
        return True
    else:
        return False


#From exercise review:
def plot_residuals(actual, predicted):
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    plt.show()     

def residuals(actual, predicted):
    return actual - predicted

def sse(actual, predicted):
    return (residuals(actual, predicted) **2).sum()

def mse(actual, predicted):
    n = actual.shape[0]
    return sse(actual, predicted) / n

def rmse(actual, predicted):
    return math.sqrt(mse(actual, predicted))

def ess(actual, predicted):
    return ((predicted - actual.mean()) ** 2).sum()

def tss(actual):
    return ((actual - actual.mean()) ** 2).sum()

def r2_score(actual, predicted):
    return ess(actual, predicted) / tss(actual)       


def regression_errors(actual, predicted):
    return pd.Series({
        'sse': sse(actual, predicted),
        'ess': ess(actual, predicted),
        'tss': tss(actual),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    })

def baseline_mean_errors(actual):
    predicted = actual.mean()
    return {
        'sse': sse(actual, predicted),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    }

def better_than_baseline(actual, predicted):
    rmse_baseline = rmse(actual, actual.mean())
    rmse_model = rmse(actual, predicted)
    return rmse_model < rmse_baseline    