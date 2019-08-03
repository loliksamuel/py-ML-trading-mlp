from buld.build_models______old import MlpTrading_old
import pandas_datareader.data as pd

# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# pd.options.display.float_format = '{:.2f}'.format

mlp_trading_old = MlpTrading_old(symbol='^GSPC')
mlp_trading_old.execute(skip_days=3600,#>400
                        modelType='mlp',#   # mlp lstm drl
                        epochs=2000,  # best 5000   or 300
                        size_hidden=15,
                        batch_size=128,#best 128
                        percent_test_split=0.33,
                        loss='binary_crossentropy',#binary_crossentropy #categorical_crossentropy
                        lr=0.00001,  # default=0.001   best=0.00001 or 0.002, for mlp, 0.0001 for lstm
                        rho=0.9,  # default=0.9   0.5 same
                        epsilon=None,
                        decay=0.0,  # 0.0 - 1.0
                        kernel_init='glorot_uniform',
                        dropout=0.2,  # 0.0 - 1.0
                        verbose = 2,  # 0, 1, 2
                        use_grid_search = False,
                        names_output = ['Green bar', 'Red Bar'],#, 'Hold Bar']#, 'Hold Bar'  #bug on adding 3rd class classify all to green
                        activation='softmax'#softmax',

                        )
