from mlp_trading_old import MlpTrading_old

mlp_trading_old = MlpTrading_old(symbol='^GSPC')
mlp_trading_old.execute(skip_days=3600,
                    modelType='mlp',# MlModel.MLP'',  # mlp lstm drl
                    epochs=50,  # best 5000
                    size_hidden=15,
                    batch_size=128,
                    percent_test_split=0.33,
                    loss='categorical_crossentropy',
                    lr=0.00001,  # default=0.001   best=0.00001 for mlp, 0.0001 for lstm
                    rho=0.9,  # default=0.9   0.5 same
                    epsilon=None,
                    decay=0.0,  # 0.0 - 1.0
                    kernel_init='glorot_uniform',
                    dropout=0.2,  # 0.0 - 1.0
                    verbose=2  # 0, 1, 2
                    )
