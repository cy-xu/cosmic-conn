python cosmic_conn/reduce_cr.py \
--data Cosmic_ConNN_datasets/LCO_NRES/banzai_frms
--snr_thres 5 \
--snr_thres_low 2.5 \
--dilation 5 \
--min_exptime 99. \
--min_cr_size 2 \
--cpus 4 \
--nres \
--comment nres

# other options

# --no_png \
# --aligned \
# --flood_fill \
# --verbose \