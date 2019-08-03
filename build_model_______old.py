from buld.build_models______old import MlpTrading_old
import pandas_datareader.data as pd

# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# pd.options.display.float_format = '{:.2f}'.format

mlp_trading_old = MlpTrading_old(symbol='^GSPC')
mlp_trading_old.execute(skip_days=13660,#>400  best=3600 #17460 #17505 rows
                        modelType='mlp',#   # mlp lstm drl
                        epochs=1000,  # best 5000   or 300
                        size_hidden=15, #best 15
                        batch_size=128,#best 128
                        percent_test_split=0.33, #best .33
                        lr=0.002,  # default=0.001   best=0.00001 or 0.002, for mlp, 0.0001 for lstm
                        rho=0.9,  # default=0.9
                        epsilon=None,#None
                        decay=0.0,  # 0.0 - 1.0
                        dropout=0.2,  # 0.0 - 1.0
                        names_output = ['Green bar', 'Red Bar'],# 'Hold Bar'],  #bug on adding 3rd class classify all to green
                        use_random_label = False,
                        use_grid_search = False,
                        kernel_init='glorot_uniform',
                        activation='softmax',#softmax',
                        loss='categorical_crossentropy',#binary_crossentropy #categorical_crossentropy
                        verbose = 2  # 0, 1, 2
                        )
