# csi_dsp.c / csi_frame.c are the same files the host tests build
# (firmware/Makefile) -- compiled here unmodified.
idf_component_register(
    SRCS "main.cc"
         "generated/model_data.cc"
         "../../wisense/csi_dsp.c"
         "../../wisense/csi_frame.c"
    INCLUDE_DIRS "." "generated" "../../wisense"
    PRIV_REQUIRES esp_timer driver esp_psram)
