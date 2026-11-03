# LinkedIn post

Attach: docs/img/dashboard.png, docs/img/clv_by_season.png

---

Can a data science model beat the betting market? I built a La Liga predictor to find out ⚽

Short answer: it can't predict results better than the market, but it can spot prices the market later corrects.

🔧 What I built
• 14 seasons of La Liga data: results, shots and odds (5,400+ matches)
• 9 models: Elo + form logistic regression, LightGBM, a Dixon-Coles goals model, a stacked ensemble, and models built on the opening odds
• Walk-forward testing: every season is predicted only by models trained on earlier seasons, never a random split
• A Streamlit dashboard that shows this week's value bets and logs paper trades

📏 How I judged it
Accuracy is the wrong metric: always picking the home team is right 45% of the time. I scored every model by log loss against the bookmakers' closing odds, the most accurate estimate there is. I judged every bet by closing line value (CLV): did it get a better price than the market's final one?

📊 Results (1,628 test matches, 2022–2026)
• No model forecasts better than the closing odds. The market is very good.
• But the bets my best model picks beat the closing price by +2.3% on average, 68% of the time. Random bets at the same prices: −0.8%.
• Profit isn't proven yet: ROI +0.3%, with a 95% interval of −12% to +13%. So I'm paper trading next.
• One model made +10% profit in the backtest while losing 3.9% to the closing odds. Profit without CLV is luck.

🧠 What I learned
1. Look at the data before modelling: I found an encoding bug, a bookmaker vanishing mid-season, and promoted teams returning with 8-year-old ratings.
2. Leakage hides everywhere, even in stacked models. Every input is built only from earlier matches, and tests check it.
3. Always compare against a control. "Positive" means nothing until you know what random gets.
4. Simple beat complex: logistic regression beat LightGBM on this small, noisy data.
5. Confidence intervals keep you honest.

Code and method: https://github.com/youssefbouktib/laliga-predictor

#DataScience #MachineLearning #Python #SportsAnalytics #Football #LaLiga
