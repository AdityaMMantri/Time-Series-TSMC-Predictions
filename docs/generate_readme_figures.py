"""Rebuild README figures from local data and saved backtest results."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
d = pd.read_csv(ROOT/'Data/TSMC_Master_Features_15Y.csv',parse_dates=['Date']).set_index('Date')
returns = d.Close.pct_change(fill_method=None)
fig,axes = plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
axes[0].plot(d.index,d.Close,color='#147D92'); axes[0].set(ylabel='Close (USD)',title='TSM: long-term growth with changing risk')
axes[1].plot(d.index,returns.rolling(21).std()*np.sqrt(252)*100,color='#8054AE',lw=.8);axes[1].set_ylabel('21-session vol (%)')
axes[2].fill_between(d.index,(d.Close/d.Close.cummax()-1)*100,0,color='#B54C58',alpha=.7);axes[2].set_ylabel('Drawdown (%)')
axes[2].set_xlabel('Observed trading dates • volatility annualized using √252 • unadjusted Close')
fig.savefig(OUT/'market_overview.png',dpi=150);plt.close(fig)
s=pd.read_csv(ROOT/'Trading/outputs/short_horizon_outputs_v2/window_summary.csv')
w=openpyxl.load_workbook(ROOT/'Trading/outputs/advanced_report/TSMC_Advanced_Report.xlsx',read_only=True,data_only=True)
rows=list(w['Summary'].values); a=pd.DataFrame([r[:13] for r in rows[5:11]],columns=rows[4][:13]);w.close()
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for ax,title,frame,modecol,modes,wincol,wins in [
    (axes[0],'Short-horizon rule search',s,'Mode',['historical_fit','prewindow_selected'],'Window',['weekly','monthly','two_months']),
    (axes[1],'Advanced adaptive strategy',a,'Mode',['Historical fit','Prewindow selected'],'Window',['Weekly','Monthly','Two months'])]:
    x=np.arange(3)
    for j,(mode,color,label) in enumerate(zip(modes,['#147D92','#B54C58'],['Historical fit','Earlier selected'])):
        v=[float(frame[(frame[modecol]==mode)&(frame[wincol]==win)].Return_pct.iloc[0]) for win in wins]
        bars=ax.bar(x+(j-.5)*.35,v,.35,label=label,color=color)
        ax.bar_label(bars,labels=[f'{z:+.2f}%' for z in v],padding=3,fontsize=9)
    ax.axhline(0,color='#627887',lw=.8);ax.set_xticks(x,['5 sessions','22 sessions','44 sessions']);ax.set(title=title,ylabel='Net return (%)',ylim=(-9,10));ax.legend(loc='upper left')
fig.suptitle('Saved tail-window results: parameter selection changes the outcome',fontsize=14)
fig.supxlabel('Independent 100,000 accounts. Historical-fit rules were selected using the reported tail.',fontsize=10)
fig.savefig(OUT/'strategy_results.png',dpi=150);plt.close(fig)
print('Created docs/assets/market_overview.png and strategy_results.png')
