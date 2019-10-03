import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict

from buld.utils import format_col_date
from data.features import ProviderDateFormat
from data.features.features import Features


class StaticDataProvider():
    _current_index = 0

    def __init__(self
                 , df       : pd.DataFrame = None
                 , csv_data_path    : str  = None
                 , do_prepare_data: bool = True
                 #, columns_map:Dict = None
                 , features_to_add ='none'
                 , **kwargs):


        self.kwargs = kwargs

        if df is not None:
            self.df = df

        elif csv_data_path is not None:
            if not os.path.isfile(csv_data_path):
                raise ValueError(
                    f'Error:  csv_data_path={csv_data_path} of StaticDataProvider, file could not be found.')
            self.df = pd.read_csv(csv_data_path)
            #features_to_add = 'none' # none(12) all(288)
            fetures = Features()
            self.df  = fetures.add_features(features_to_add, self.df)

            #formatted = reduce_mem_usage        (formatted)
            #self.print_is_stationary(self.df)
            #print(f'prepared_data={self.df}\ndescribe=\n{self.df.describe()}')
            #self.plot_stats()


        else:
            raise ValueError(
                'Error: StaticDataProvider requires either a "data_frame" or "csv_data_path argument".')

        if do_prepare_data:
            #self.columns = self.df.columns
            #self.df = self.data_prepare(self.df)
            self.df = format_col_date(self.df)
            #self.df = self._sort_by_date      (self.df )
            if isinstance(self.df,  pd.DataFrame):
                d = csv_data_path.replace('.csv', '_with_features.csv')
                self.df.to_csv(d, index=False )
                self.logger.info(f'saved file {d}')


        print(f'(n_samples, n_features) = {self.df.shape}')

        self.columns = self.df.columns





    @staticmethod
    def from_prepared(df: pd.DataFrame, date_format: ProviderDateFormat, **kwargs):
        return StaticDataProvider(date_format=date_format, df=df, csv_data_path=None, do_prepare_data=False, **kwargs)


    def split_data_train_test(self, train_split_percentage: float = 0.8) :
        len_train = int(train_split_percentage * len(self.df))

        train_df = self.df[:len_train].copy()
        test_df  = self.df[len_train:].copy()

        train_provider = StaticDataProvider.from_prepared(df=train_df, date_format=self.date_format, **self.kwargs)
        test_provider  = StaticDataProvider.from_prepared(df=test_df , date_format=self.date_format, **self.kwargs)

        return train_provider, test_provider


    def get_all_historical_obs(self) -> pd.DataFrame:
        return self.df


    def has_next_obs(self) -> bool:
        return self._current_index < len(self.df)


    def reset_obs_index(self) -> int:
        self._current_index = 0


    def next_obs(self) -> pd.DataFrame:
        frame = self.df[self.columns].values[self._current_index]
        frame = pd.DataFrame([frame], columns=self.columns)

        self._current_index += 1

        return frame
