"""Data access layer.

Loads forecast/observation fields (synthetic arrays today, real gridded
NetCDF/HDF products later) and normalizes them into the array shapes the
verification package expects. Owns all file-format and I/O concerns.
"""
