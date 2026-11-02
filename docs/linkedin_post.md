# LinkedIn post (draft)

I built a La Liga match predictor to answer one question: can a data science model find mispriced odds?

The honest answer: it can't predict results better than the market, but it can spot prices the market later corrects. Here's what I learned.

⚽ The setup
• 14 seasons of La Liga results, odds and shots (5,400+ matches)
• 9 models: Elo + form logistic regression, LightGBM, a Dixon-Coles Poisson model, a stacked ensemble, and market-aware models built on the opening odds
• Walk-forward testing only: every season is predicted by models trained on earlier seasons

📏 The benchmark
Accuracy is the wrong metric: always picking the home team is right 45% of the time. I judged every model by log loss against the sharp closing odds, and every bet by closing line value (CLV): did it get a better price than the market's final, best-informed opinion?

📊 What I found (1,628 test matches, 2022–2026)
• No model forecasts better than the closing line. My best football-only model (log loss 0.973) is well behind it (0.956).
• The opening market is almost as good as the close. Adding my football features to it didn't help on average.
• But combining a market-anchored model with line shopping picks bets that beat the closing line by +2.3% on average, 68% of the time. Random bets at the same prices: −0.8%.
• Profit isn't proven: ROI is +0.3% with a 95% interval of −12% to +13%. That's why I'm paper trading next.
• One model showed +10% profit in the backtest while losing 3.9% to the closing line. Profit without CLV is luck.

🧠 What I learned
1. Look at the data before modelling. The first real-data run found an encoding bug, a bookmaker vanishing mid-season, and promoted teams returning with 8-year-old ratings.
2. Leakage hides everywhere, including in stacked models: every model input is built only from earlier matches.
3. Always compare with a control. "Positive CLV" means nothing until you know what random bets get.
4. Confidence intervals keep you honest. A 600-bet backtest can't prove profit.
5. I used AI agents as a separate tester and fixer: 140 tests, 10 real bugs found and fixed.

Code, charts and method: [GitHub link]

#DataScience #MachineLearning #Python #SportsAnalytics #Football #LaLiga
