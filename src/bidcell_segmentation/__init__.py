from bidcell import BIDCellModel
BIDCellModel.get_example_data()
model = BIDCellModel("params_small_example.yaml")
model.run_pipeline()