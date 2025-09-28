from src.features import build_patients_data
from src.logging_config import setup_logger

logger = setup_logger(verbose=True)


def main():
    logger.info("🚀 Starting main pipeline")
    # dataset.process_prescription_files()  # call function from dataset.py
    # plots.plot_all_patients_in_parallel()
    build_patients_data()
    logger.success("✅ Pipeline finished successfully")


if __name__ == "__main__":
    main()
    # ! SpiroUtils integrates the patient’s demographic information (age, gender, height) to calculate the Predicted Value and Z-score for these metrics based on the multi-ethnic reference equations published by the Global Lung Function Initiative in 2012. [Quanjer, P.H., Stanojevic, S., Cole, T.J., Baur, X., Hall, G.L., Culver, B.H., Enright, P.L., Hankinson, J.L., Ip, M.S., Zheng, J. et al. (2012). Multi-ethnic reference values for spirometry for the 3–95-yr age range: the global lung function 2012 equations. European Respiratory Society.]
    # !    Compute Pred, %Pred, z-scores via GLI-2012 (sex, age, height) for FEV₁/FVC/FEF25–75.
    # !    Retrieve GOLD rules relevant to each case (diagnosis threshold, severity).
    # !Render a JSON-structured clinical note as target text.
    # * Find which variable containts most zero  values
    # * We may group the variables with measurements
    # * SHAP analysis
    # * Make a plot showing duration of patient data (first to last record) and how many records
    # * Determine target variable (FEV1/FVC ratio or Gold Staging)
    # * How to explain in the dataset two consiqitive FEV1 values have uneven time intervals? Maybe years apart. And we don't want to use interpolation.

    # ! Deep research
    # ? Importantly, irregular intervals between PFTs should be accounted for – e.g. by including time-stamps or using methods for irregular time-series.
    # ? In cases of relatively few time points, one can engineer features like “FEV₁ decline per year” as inputs to simpler models(like mixed-effects regression to estimate each patient’s FEV₁ slope and then used slopes in a predictive algorithm).
    # ? Labels and Prediction Targets - Future COPD diagnosis (yes/no), Regression targets (FEV₁ decline rate (mL/year) or future FEV₁ % predicted, FEV₁/FVC ratio 5 years from now), Time-to-event (survival analysis - survival neural networks or simpler Cox models with learned risk scores)
    # ? Handling Confounding Factors (Age, Sex - percent-predicted). Smoking - Pack-years (a cumulative measure) or duration of smoking and current status (current vs former vs never) / indirect proxies of smoking (like low DLCO or high decline rate).
    # ? Integrating Demographic Data with PFT Trends - Feature Augmentation / Multi-Branch Model
    # ? Data normalization is important too. Continuous features like age and pack-years should be scaled (the model shouldn’t be thrown off by age being, say, “60” while FEV₁ is “1.2” liters – different scales)
    # ? Confounder checks- Check performance across age groups or sexes to confirm it’s truly picking up disease signals. One could perform ablation experiments (train model without age or without smoking) to see how much performance drops – if it drops a lot, that feature was carrying important signal.
    # ? Class imbalance - Techniques like oversampling, focal loss, or stratified batch sampling can be used so that the model doesn’t simply always predict “no risk”. Also, evaluation metrics like AUC or precision-recall are more informative than accuracy in such imbalanced scenarios.
    # ? Interpretable features
    # ? A successful model might flag, say, a 45-year-old smoker with FEV₁ falling from 95% to 80% predicted over 5 years, even though at the last measurement FEV₁/FVC is 0.72 (technically normal). If the model is working, such a person should be assigned high risk – which aligns with clinical intuition and the literature.
