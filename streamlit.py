import streamlit as st
import numpy as np

st.write(f"## Effect of ML precision & recall on naiive occupancy")

# inputs
precision = st.number_input(
    "Classifier precision (one clip)", value=0.99, step=0.0001, format="%0.4f"
)
recall = st.number_input(
    "Classifier recall (one clip)", value=0.5, step=0.0001, format="%0.4f"
)
availability = st.number_input(
    "Availability (fraction of clips at occupied site with detectable cue)",
    value=0.02,
    step=0.000001,
    format="%0.6f",
)
sites = st.slider("Number of sites", 1, 1000, 100, step=1)
clips = int(st.slider("Number of clips per site", 1e3, 1e6, 1e5, step=1e3))
true_occupancy = st.slider("True Occupancy", 0.0, 1.0, 0.5)

# calculations
n_occ_sites = int(sites * true_occupancy)
n_empty_sites = sites - n_occ_sites
expected_n_false_positives_per_empty_site = clips * (1 - precision)
chance_of_fp_at_site = 1 - precision**clips
expected_n_sites_with_false_positive = np.round(chance_of_fp_at_site * n_empty_sites)
n_clips_with_positive = int(clips * availability)  # n clips with cue at occupied site
# chance of no true positive and no false positive at occupied site
no_tp_occ = (1 - recall) ** n_clips_with_positive
no_fp_occ = (precision) ** (clips - n_clips_with_positive)
expected_n_occ_sites_with_positive = int(n_occ_sites * (1 - no_tp_occ * no_fp_occ))

# site recall is 1- (chance of never detecting it on any of the clips)
site_level_recall = 1 - (1 - recall) ** int(clips * availability)

# effects of both fp and fn on naive occupancy at site level
expected_naive_occupancy_w_fp = (
    expected_n_occ_sites_with_positive + expected_n_sites_with_false_positive
) / sites

# outputs
st.write(
    f"Expected n false positives per empty site: {expected_n_false_positives_per_empty_site:0.2f}"
)
st.write(f"Chance of 1+ false positive clips at a site: {chance_of_fp_at_site:0.4f}")
st.write(f"Site-level recall: {site_level_recall:0.3f}")
st.write(
    f"Site-level naiive occupancy with false positives: {expected_naive_occupancy_w_fp:0.4f}"
)
