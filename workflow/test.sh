source workflow/workflow_env
python -m doe_xstock \
    -d $DOE_XSTOCK_DATABASE_FILEPATH \
        -u "/Users/kingsleyenweye/Library/CloudStorage/Box-Box/My Documents/large_data/doe_xstock/simulation_output" \
            -l "/Users/kingsleyenweye/Library/CloudStorage/Box-Box/My Documents/large_data/doe_xstock/lstm_train_data" \
                -i $DOE_XSTOCK_IDD_FILEPATH \
                    dataset \
                        $DOE_XSTOCK_INSERT_DATASET_TYPE \
                            $DOE_XSTOCK_INSERT_WEATHER_DATA \
                                $DOE_XSTOCK_INSERT_YEAR_OF_PUBLICATION \
                                    $DOE_XSTOCK_INSERT_RELEASE \
                                        set_lstm_train_data \
                                            281166 \
                                                -s $DOE_XSTOCK_DEFAULT_SEED \
                                                    -r $DOE_XSTOCK_LSTM_PARTIAL_LOAD_SIMULATION_ITERATIONS