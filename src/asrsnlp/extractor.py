"""Data Extractor from source using func in .apifunc
"""

from apifunc import dsdownloader
dsdownloader(owner="drxc75", dataset="nasa-asrs",
             output="/home/ensai/dev/asrsnlp/data/02_intermediate")