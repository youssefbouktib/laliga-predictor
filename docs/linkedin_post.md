# LinkedIn post (draft)

I built a La Liga match predictor to answer one question: can a data science model beat the betting market?

Short answer: not yet. Here is what I learned building it.

⚽ The setup
• 14 seasons of La Liga results and odds (5,400+ matches)
• 3 models: logistic regression on Elo ratings and form, LightGBM, and a Dixon-Coles Poisson model with time decay
• Walk-forward testing only: every season is predicted by a model trained on earlier seasons, never a random split

📏 The benchmark
Accuracy is the wrong metric here. A model that always picks the home team is right 45% of the time. What matters is whether my probabilities beat the bookmaker's, so I compared every model against the margin-free closing odds using log loss, and measured closing line value (CLV) on every bet.

📊 The results (1,628 test matches, 2022–2026)
• Best model (Dixon-Coles): log loss 0.976
• Bookmaker closing odds: 0.956
• The model's "value bets" beat the closing price only 41% of the time, with flat-stake ROI around −17%
• Quarter-Kelly staking turned a bankroll of 100 into about 4. Kelly amplifies your edge, including a negative one.

🧠 What I learned
1. Look at the data before modelling it. My first real-data run found a file encoding bug, a bookmaker (Pinnacle) disappearing from the data mid-season, and promoted teams returning after 8 years with stale ratings.
2. Leakage is easy to create and hard to see. Every feature is computed in one chronological pass that only sees the past.
3. Compare like with like. CLV is meaningless if your bet price and the closing price come from different bookmakers.
4. A well-calibrated model can still be worse than the market. Calibration and edge are different things.
5. Honest negative results are results. The gap to the market is now a number I can try to close.

Next: adding expected goals (xG) features and testing whether the model adds anything to the opening market price.

Code, charts and method: [GitHub link]

#DataScience #MachineLearning #Python #SportsAnalytics #Football #LaLiga
