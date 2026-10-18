"""Independent replay of Top 10/20/50, including distributions, for the 2020-2026 audit.
Run from any directory; writes checks under factor_engine/target/strategy_audit/corrected.
"""
from pathlib import Path
import pandas as pd, numpy as np, json, tomllib
root=Path(__file__).resolve().parents[2]; data=Path(tomllib.loads((root/'config.toml').read_text(encoding='utf-8'))['paths']['base_data_dir']);audit=root/'factor_engine/target/strategy_audit/corrected';audit.mkdir(exist_ok=True)
frames={n:pd.read_parquet(data/f'strategy/stock/strategy_001_top{n}_5d/holdings.parquet') for n in (10,20,50)}
events=pd.concat([pd.read_parquet(p) for p in (data/'chn_stock_data/dividend').glob('*.parquet')],ignore_index=True)
events=events.loc[events.div_proc.eq('实施') & events.record_date.ge(20200101) & events.record_date.le(20260424) & events.ex_date.gt(events.record_date)].drop_duplicates(['ts_code','record_date','ex_date'])
by_record={int(d):f.to_dict('records') for d,f in events.groupby('record_date')}
states={n:dict(cash=100000.,q={},mark={},pending=[],receivables=[],dividend_income=0.,max_receivable_error=0.,max_cash_error=0.,max_equity_error=0.,max_quantity_error=0.,price_errors=0,concentrations=[],gaps=[]) for n in frames}
for idx,date in enumerate(frames[10].trade_date):
 m=pd.read_parquet(data/f'chn_stock_data/daily/pv/{date//10000}/{date}.parquet').set_index('ts_code').to_dict('index')
 for n,f in frames.items():
  s=states[n]; row=f.iloc[idx];q=s['q'];mark=s['mark']; eq_before=s['cash']+sum(v*mark[k] for k,v in q.items())+sum(v for _,v in s['receivables']);gap=0
  for pay,amount in s['receivables']:
   if pay<=date:s['cash']+=amount
  s['receivables']=[(pay,amount) for pay,amount in s['receivables'] if pay>date]
  for event,qty in s['pending']:
   if event['ex_date']>date:continue
   sym=event['ts_code'];rate=event['stk_div'];rate=round(rate,6) if np.isfinite(rate) and rate>0 else 0
   bonus=np.floor(qty*rate+1e-6)
   if bonus:q[sym]=q.get(sym,0)+bonus
   cashrate=event['cash_div_tax'];cashrate=round(cashrate,6) if np.isfinite(cashrate) and cashrate>0 else 0
   amount=qty*cashrate;s['dividend_income']+=amount
   pay=event['pay_date'];pay=pay if np.isfinite(pay) and pay>=event['ex_date'] else event['ex_date']
   if pay<=date:s['cash']+=amount
   else:s['receivables'].append((pay,amount))
  s['pending']=[(event,qty) for event,qty in s['pending'] if event['ex_date']>date]
  for sym,qty in q.items():
   if sym in m and np.isfinite(m[sym]['pre_close']):
    diff=m[sym]['pre_close']-mark[sym]
    if abs(diff)>0.011:
     s['gaps'].append(dict(date=int(date),symbol=sym,quantity=qty,previous_mark=mark[sym],pre_close=m[sym]['pre_close'],gap_value=qty*diff))
  for sym,qty,price in zip(json.loads(row.trade_symbols_json),json.loads(row.trade_signed_quantities_json),json.loads(row.trade_prices_json)):
   actual=m[sym]['open'];s['price_errors']+=int(abs(actual-price)>.00501)
   s['cash']-=qty*actual+abs(qty)*actual*(.0007 if qty>0 else .0012)
   q[sym]=q.get(sym,0)+qty;mark[sym]=actual
   if abs(q[sym])<1e-8:q.pop(sym)
  for sym in q:
   if sym in m and np.isfinite(m[sym]['close']) and m[sym]['close']>0:mark[sym]=m[sym]['close']
  equity=s['cash']+sum(v*mark[k] for k,v in q.items())+sum(v for _,v in s['receivables']);s['max_receivable_error']=max(s['max_receivable_error'],abs(sum(v for _,v in s['receivables'])-row.dividend_receivable));s['max_cash_error']=max(s['max_cash_error'],abs(row.cash-s['cash']));s['max_equity_error']=max(s['max_equity_error'],abs(equity-100000-row.account_pnl))
  saved=dict(zip(json.loads(row.symbols_json),json.loads(row.signed_quantities_json)))
  s['max_quantity_error']=max(s['max_quantity_error'],max((abs(saved.get(k,0)-q.get(k,0)) for k in saved.keys()|q.keys()),default=0))
  for event in by_record.get(int(date),[]):
   qty=q.get(event['ts_code'],0)
   if qty>0:s['pending'].append((event,qty))
  biggest=max(q,key=lambda k:q[k]*mark[k],default=None)
  s['concentrations'].append(dict(date=int(date),symbol=biggest,weight=q[biggest]*mark[biggest]/equity if biggest else 0,equity=equity,daily_return=equity/eq_before-1))
 if idx%400==0:print('Replay day',idx,flush=True)
summary=[]
for n,s in states.items():
 pd.DataFrame(s.pop('gaps')).to_csv(audit/f'top{n}_preclose_gaps.csv',index=False)
 cs=pd.DataFrame(s.pop('concentrations'));cs.to_csv(audit/f'top{n}_concentration.csv',index=False)
 s.pop('q');s.pop('mark');s.pop('pending');s.pop('receivables');s['top_n']=n;summary.append(s)
 print(n,s,'maxweight',cs.nlargest(1,'weight').to_dict('records'),'worstday',cs.nsmallest(1,'daily_return').to_dict('records'),flush=True)
(audit/'replay_summary.json').write_text(json.dumps(summary,indent=2))

for result in summary:
    assert result["max_cash_error"] <= .01 and result["max_equity_error"] <= .01
    assert result["max_quantity_error"] == 0 and result["price_errors"] == 0
    assert result["max_receivable_error"] <= .01
