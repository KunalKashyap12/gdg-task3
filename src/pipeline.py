"""
End-to-end training and evaluation orchestration pipeline for the News Classification System.
Executes data loading, basic and advanced preprocessing, feature extraction,
multi-model benchmarking, champion model training, test dataset evaluation,
visualization generation, and artifact serialization.
"""

import sys
import time
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from src.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    ARTIFACTS_DIR,
    MODEL_PATH,
    METRICS_PATH,
    CONFUSION_MATRIX_PATH,
    PER_CLASS_METRICS_PATH,
    TOP_WORDS_PATH,
    LABEL_MAP,
    CLASS_NAMES,
    RANDOM_STATE,
    TFIDF_MAX_FEATURES
)
from src.preprocessing import TextPreprocessor
from src.features import get_word_tfidf_vectorizer
from src.models import get_available_models, create_classification_pipeline
from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_per_class_metrics,
    get_top_features_per_class,
    perform_error_analysis
)


def run_pipeline(sample_size: int = None, use_advanced_clean: bool = False):
    """
    Executes the complete machine learning lifecycle.
    """
    print("=" * 80)
    print("        AG NEWS 4-CLASS NLP CLASSIFICATION SYSTEM")
    print("=" * 80)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Load Datasets
    # -------------------------------------------------------------
    print(f"\n[1/7] Loading datasets...")
    t0 = time.time()
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    if sample_size and sample_size < len(train_df):
        print(f"Sampling {sample_size:,} records from training data for faster iteration...")
        train_df = train_df.groupby("Class Index", group_keys=False).apply(
            lambda x: x.sample(n=sample_size // 4, random_state=RANDOM_STATE)
        ).reset_index(drop=True)

    print(f"Loaded Train set: {len(train_df):,} samples | Test set: {len(test_df):,} samples ({time.time()-t0:.2f}s)")
    print("Class Distribution in Training Set:")
    for cid, cname in LABEL_MAP.items():
        cnt = (train_df["Class Index"] == cid).sum()
        pct = cnt / len(train_df) * 100
        print(f"  - Class {cid} ({cname:8s}): {cnt:6,} samples ({pct:.1f}%)")

    # -------------------------------------------------------------
    # Step 2: Text Preprocessing
    # -------------------------------------------------------------
    print(f"\n[2/7] Preprocessing text (Basic & Advanced Text Cleaning)...")
    t0 = time.time()
    preprocessor = TextPreprocessor(
        use_lemmatization=use_advanced_clean,
        remove_stopwords=use_advanced_clean,
        expand_contractions=True,
        title_weight=2  # Title given double weight for higher topical density
    )

    clean_fn = preprocessor.clean_advanced if use_advanced_clean else preprocessor.clean_basic
    print(f"Applying {'Advanced (Lemmatization + Stopwords)' if use_advanced_clean else 'Basic (HTML, URLs, Contractions, Boilerplate)'} text cleaning...")

    # Combine Title and Description with title emphasis
    raw_train_combined = preprocessor.combine_title_description(train_df["Title"], train_df["Description"])
    raw_test_combined = preprocessor.combine_title_description(test_df["Title"], test_df["Description"])

    train_cleaned = raw_train_combined.apply(clean_fn)
    test_cleaned = raw_test_combined.apply(clean_fn)
    print(f"Preprocessing completed in {time.time()-t0:.2f}s")

    y_train = train_df["Class Index"].values
    y_test = test_df["Class Index"].values

    # -------------------------------------------------------------
    # Step 3: Feature Extraction (TF-IDF Vectorization)
    # -------------------------------------------------------------
    print(f"\n[3/7] Feature Extraction (Word & Bigram TF-IDF)...")
    t0 = time.time()
    vectorizer = get_word_tfidf_vectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(train_cleaned)
    X_test_vec = vectorizer.transform(test_cleaned)
    print(f"Extracted TF-IDF Feature Matrix: {X_train_vec.shape[0]:,} docs x {X_train_vec.shape[1]:,} n-grams ({time.time()-t0:.2f}s)")

    # -------------------------------------------------------------
    # Step 4: Multi-Model Benchmark & Comparison
    # -------------------------------------------------------------
    print(f"\n[4/7] Benchmarking candidate NLP models on Test set...")
    models = get_available_models(random_state=RANDOM_STATE)
    candidate_keys = ["LinearSVC", "LogisticRegression"]

    results = []
    trained_models = {}

    for name in candidate_keys:
        model = models[name]
        t_start = time.time()
        model.fit(X_train_vec, y_train)
        fit_time = time.time() - t_start

        # Inference speed test
        t_inf_start = time.time()
        preds = model.predict(X_test_vec)
        inf_time = time.time() - t_inf_start
        throughput = len(y_test) / max(inf_time, 1e-5)

        m = calculate_metrics(y_test, preds)
        results.append({
            "Model": name,
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "Macro F1 (%)": round(m["macro_f1"] * 100, 2),
            "Weighted F1 (%)": round(m["weighted_f1"] * 100, 2),
            "Training Time (s)": round(fit_time, 2),
            "Throughput (docs/s)": int(throughput)
        })
        trained_models[name] = model

    # Display comparison table
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Table:")
    print("-" * 75)
    print(results_df.to_string(index=False))
    print("-" * 75)

    # -------------------------------------------------------------
    # Step 5: Train Champion Calibrated Pipeline
    # -------------------------------------------------------------
    print(f"\n[5/7] Training Champion Calibrated Pipeline (LinearSVC + Platt Probability Calibration)...")
    champion_pipeline = create_classification_pipeline(
        include_char_ngrams=False,
        word_max_features=TFIDF_MAX_FEATURES
    )
    t0 = time.time()
    champion_pipeline.fit(train_cleaned, y_train)
    print(f"Champion Pipeline trained and calibrated in {time.time()-t0:.2f}s")

    # -------------------------------------------------------------
    # Step 6: Test on the Provided Test Dataset
    # -------------------------------------------------------------
    print(f"\n[6/7] Comprehensive Evaluation on Test Dataset ({len(test_df):,} samples)...")
    y_test_pred = champion_pipeline.predict(test_cleaned)
    final_metrics = calculate_metrics(y_test, y_test_pred)

    print("\n" + "=" * 50)
    print(f" FINAL TEST ACCURACY: {final_metrics['accuracy']*100:.2f}%")
    print(f" MACRO F1-SCORE:      {final_metrics['macro_f1']*100:.2f}%")
    print(f" WEIGHTED F1-SCORE:   {final_metrics['weighted_f1']*100:.2f}%")
    print("=" * 50)

    print("\nPer-Class Breakdown:")
    per_class_table = []
    for cls_name, vals in final_metrics["per_class"].items():
        per_class_table.append({
            "Category": cls_name,
            "Precision (%)": f"{vals['precision']*100:.2f}%",
            "Recall (%)": f"{vals['recall']*100:.2f}%",
            "F1-Score (%)": f"{vals['f1_score']*100:.2f}%",
            "Support": f"{vals['support']:,}"
        })
    per_class_df = pd.DataFrame(per_class_table)
    print("-" * 55)
    print(per_class_df.to_string(index=False))
    print("-" * 55)

    # Save metrics JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"\nEvaluation metrics saved to {METRICS_PATH}")

    # Generate & Save Visualizations
    plot_confusion_matrix(y_test, y_test_pred, save_path=CONFUSION_MATRIX_PATH)
    plot_per_class_metrics(final_metrics, save_path=PER_CLASS_METRICS_PATH)

    # Top Informative Features
    base_linear_model = trained_models["LinearSVC"]
    top_features = get_top_features_per_class(
        vectorizer,
        base_linear_model,
        top_n=12,
        save_path=TOP_WORDS_PATH
    )

    print("\nTop 5 Most Predictive Keywords Per Category:")
    for cls_name, words in top_features.items():
        print(f"  - {cls_name:8s}: {', '.join(words[:5])}")

    # Error Analysis
    print("\nError Analysis (Top Misclassification Confusion Pairs):")
    error_summary = perform_error_analysis(test_df, y_test, y_test_pred, max_examples_per_pair=1)
    for item in error_summary:
        print(f"\n* {item['pair']} (Count: {item['misclassified_count']})")
        if item["examples"]:
            ex = item["examples"][0]
            print(f"   Example Title: \"{ex['title']}\"")
            print(f"   Snippet: \"{ex['description'][:100]}...\"")

    # Serialize champion pipeline artifact
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(champion_pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nChampion model pipeline serialized to {MODEL_PATH}")

    # -------------------------------------------------------------
    # Step 7: Real-world Sample Inference Demonstrations
    # -------------------------------------------------------------
    print("\n[7/7] Demonstrating Inference on Diverse Sample News Articles...")
    sample_articles = [
        {
            "category": "World",
            "title": "United Nations Security Council Passes Historic Ceasefire Resolution",
            "description": "Diplomats from 15 nations reached consensus in Geneva regarding immediate peacekeeping deployment and humanitarian corridors in conflict zones."
        },
        {
            "category": "Sports",
            "title": "Real Madrid Scores Stoppage-Time Thriller to Secure Champions League Quarterfinals",
            "description": "A stunning curling header in the 94th minute silenced the home crowd as the star striker notched his 25th goal of the campaign."
        },
        {
            "category": "Business",
            "title": "Federal Reserve Holds Interest Rates Steady as Inflation Cools and Markets Surge",
            "description": "Wall Street indices rallied following the central bank chairman's press conference signaling potential rate cuts and resilient quarterly corporate earnings."
        },
        {
            "category": "Sci/Tech",
            "title": "NASA James Webb Space Telescope Discovers Oldest Galaxy Candidates in Deep Cosmic Survey",
            "description": "Astronomers utilized infrared spectrometry algorithms to detect spectral redshifts from galaxies formed merely 300 million years after the Big Bang."
        }
    ]

    for sample in sample_articles:
        combined = preprocessor.clean_basic(
            preprocessor.combine_title_description(sample["title"], sample["description"])
        )
        pred_id = champion_pipeline.predict([combined])[0]
        probs = champion_pipeline.predict_proba([combined])[0]

        print(f"\nTarget: [{sample['category']}] -> Predicted: [{LABEL_MAP[pred_id]}] (Confidence: {np.max(probs)*100:.1f}%)")
        print(f"Title: \"{sample['title']}\"")
        prob_str = " | ".join([f"{CLASS_NAMES[i]}: {probs[i]*100:.1f}%" for i in range(4)])
        print(f"Confidence Distribution: {prob_str}")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in: {ARTIFACTS_DIR.resolve()}")
    print("=" * 80)
