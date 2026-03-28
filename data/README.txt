# Place your penguins.csv file here.
#
# Expected format (no header required, but supported):
#   feature1,feature2,feature3,feature4,feature5,class_label
#
# - 5 numeric features per row
# - class_label: integer 0, 1, or 2  (adjust in load_data() if yours uses 1,2,3)
# - 50 rows per class = 150 rows total
#
# load_data() in nn_core.py will split automatically:
#   first 30 rows per class  → training
#   last  20 rows per class  → testing
