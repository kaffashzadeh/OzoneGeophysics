"""
This program is written to arrange (harmonize) the messy data time series.
"""
# imports python standard libraries
import os
import sys
import inspect
import glob
from time import strptime
from dateparser.calendars.jalali import JalaliCalendar
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

# import local libraries
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir + '/../../')
os.sys.path.insert(0, parentdir)

__author__='Najmeh Kaffashzadeh'
__author_email__ = 'n.kaffashzadeh@ut.ac.ir'


class HarmonizeDataSeries:

    name = 'Harmonize data time series'

    def __init__(self, vars1=None, vars2=None, vars3=None, s_year=2007, e_year=2021):
        """
        This initializes the variables.

        Args:
            vars1(str or list): variable names
            vars2(str or list): variables name
            vars3(str or list): variables name
            s_year(int): the start (first) year
            e_year(int): the end (last) year

        Note:
              At some data files, the variables are stored with the different names.
        """
        self.vars1 = vars1
        self.vars2 = vars2
        self.vars3 = vars3
        self.f_year = s_year
        self.l_year = e_year
        self.c_month = None  # current month
        self.c_year =  None  # current year
        self.df = pd.DataFrame()  # data series
        self.path = None          # path of the data

    def reverse_day_month(self, df=None):
        """
        It reverses day and month.

        Args:
            df(pd.DataFrame): data data series

        Returns:
            df_new(pd.DataFrame): modified (corrected) time series

        Note:
            In some data files, the format of month and day are reversed. As an data,
            01.12.2007 (1 December 2007) is stored as 12.01.2007 (12 January 2007).

        Ref:
            https://stackoverflow.com/questions/50367656/python-pandas-pandas-to-datetime-
            is-switching-day-month-when-day-is-less-t
        """
        # Add a few columns for date-time information
        df['tmp'] = pd.to_datetime(df.index, unit='s')
        df['tmp'] = df['tmp'].apply(lambda x: x.replace(microsecond=0))
        df['date'] = [d.date() for d in df['tmp']]
        df['time'] = [d.time() for d in df['tmp']]
        df[['year', 'month', 'day']] = df['date'].apply(lambda x: pd.Series(x.strftime("%Y-%m-%d").split("-")))
        df['day'] = pd.to_numeric(df['day'], errors='coerce')
        df['month'] = pd.to_numeric(df['month'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        # Loop to look for days less than 13 and then swap the day and month
        for index, d in enumerate(df['day']):
            if (d < 13):
                df.day.iloc[index], df.month.iloc[index] = df.month.iloc[index], df.day.iloc[index]
        # Convert series to string type in order to merge them
        df['day'] = df['day'].astype(str)
        df['month'] = df['month'].astype(str)
        df['year'] = df['year'].astype(str)
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df['date'] = df['date'].astype(str)
        df['time'] = df['time'].astype(str)
        # Merge time and date and place result in our column
        df['tmp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df_new = df.set_index(df['tmp'])
        df_new.drop(df_new[['date', 'year', 'month', 'day', 'time', 'tmp']],
                           axis=1, inplace=True)
        return df_new

    def correct_time(self, df=None):
        """
        It corrects time (hour).

        Args:
            df(pd.DataFrame): data data series

        Returns:
            df_new(pd.DataFrame): modified (corrected) time series

        Note:
            For some data series, at 00 when time series is shifted to the next month.
            In fact, it wants to shift to the next day, but since the day and month are reversed,
            it shifts to the next month.
            For instance, for the day 13 of month Dec, it is written as 12.13.2007 00:00:00. While
            there is no month larger than or equal to 12.
        """
        # Add a few columns for date-time information
        df['tmp'] = pd.to_datetime(df.index, unit='s')
        df['tmp'] = df['tmp'].apply(lambda x: x.replace(microsecond=0))
        df['date'] = [d.date() for d in df['tmp']]
        df['time'] = [d.time() for d in df['tmp']]
        df[['year', 'month', 'day']] = df['date'].apply(lambda x: pd.Series(x.strftime("%Y-%m-%d").split("-")))
        df['day'] = pd.to_numeric(df['day'], errors='coerce')
        df['month'] = pd.to_numeric(df['month'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        # Loop to look for hour 00 and day 13
        for index, t in enumerate(df['time']):
            if (t.hour == 0) and (df.month.iloc[index] == self.c_month+1):
                try:
                    df.day.iloc[index], df.month.iloc[index] = df.day.iloc[index + 1], df.month.iloc[index + 1]
                except IndexError:
                    df.day.iloc[index], df.month.iloc[index] = df.day.iloc[index], df.month.iloc[index]
            if (t.hour == 0) and (self.c_month==12) and (df.day.iloc[index] == 13): #(df.month.iloc[index] == df.index.month[0] + 1):
                df.day.iloc[index], df.month.iloc[index] = df.day.iloc[index + 1], df.month.iloc[index + 1]
        # Convert series to string type in order to merge them
        df['day'] = df['day'].astype(str)
        df['month'] = df['month'].astype(str)
        df['year'] = df['year'].astype(str)
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df['date'] = df['date'].astype(str)
        df['time'] = df['time'].astype(str)
        # Merge time and date and place result in our column
        df['tmp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df_new = df.set_index(df['tmp'])
        df_new.drop(df_new[['date', 'year', 'month', 'day', 'time', 'tmp']],
                    axis=1, inplace=True)
        return df_new

    def convert_jalali_to_gregorian(self, df=None):
        """"
        It corrects time (hour).

        Args:
            df(pd.DataFrame): data data series

        Returns:
               df_new(pd.DataFrame): a new time series with gregorian date-time index
        """
        df['jalali'] = df.index
        df['gregorian'] = df['jalali'].apply(lambda x: JalaliCalendar(str(x)).get_date().date_obj
                                            if pd.notnull(x) else x)
        exit()
        df.set_index(df['gregorian'], inplace=True, drop=True)
        df_new = df.drop(['jalali','gregorian'], axis=1)
        return df_new # df.drop(['jalali','gregorian'], axis=1, inplace=True)

    def read_check_raw_data(self):
        """
        It reads the raw data.
        """
        # Read all files in the given path
        for file_name in glob.glob(self.path + '/*.xls'):
            # file_name = '/Users/nkaffash/Documents/PostDoc2/Programming/src/../../Data/PollutantTehran/Geophysic/y2017/Oct 2017.xls'
            head_tail = os.path.split(file_name)
            month_name = str(head_tail[1][0:3])
            self.c_month = strptime(str(month_name),'%b').tm_mon
            print('Current file name is ' + file_name)
            try:
                df = pd.read_excel(file_name, sheet_name='Sheet1',
                                   index_col=0,
                                   skiprows=[1],
                                   parse_dates=True)
                                   # date_parser=JalaliCalendar,
                                   # skipfooter=3,
                                   # comment='OK')
            except ValueError:
                print('There is an error in the file!')
            idx_f = pd.Series.first_valid_index(df)
            try:
                idx_f_y = idx_f.year
            except:
                idx_f_y = int(idx_f[0:4])
            if idx_f_y < 1500:
                print('Date-time is recorded based on the Jalali calender!')
                df = self.convert_jalali_to_gregorian(df=df)
                print('Date-time was converted to the Gregorian calender!')
            else:
                pass
            df.index = pd.to_datetime(df.index, dayfirst=True, errors='coerce')
            df = df.loc[str(self.c_year)]
            if self.c_year < 2011:
                # after 2011 the date time are written in the correct format,
                # the files contain all months
                df_m1 = self.reverse_day_month(df=df)  # first modified
                df_m2 = self.correct_time(df=df_m1)    # second modified
            else:
                df_m2=df
            print(df_m2.head())
            df_m2.to_csv(file_name.split(".xls",1)[0]+'.csv') # save data of that month
            # df.index = pd.to_datetime(df.index, format='%d-%m-%yyyy %H:%M:%S')
            df_new = df_m2.loc[str(self.c_year) + '-' + str(self.c_month)]  # select only data of the month
            print(self.c_month)
            df_sel = self.select_data(df=df_new)
            print(df_sel.head(4))
            self.harmonize_labels(df=df_sel)
            self.concat_data(df=df_sel)

    def select_data(self, df=None):
        """
        It selects the data columns based on the selected variables.

        Args:
            df(pd.DataFrame): data series

        Returns:
              df(pd.DataFrame): selected data series
        """
        try:
            return df[self.vars1]
        except KeyError:
            try:
                return df[self.vars2]
            except KeyError:
                return df[self.vars3]

    def harmonize_labels(self, df=None):
        """
        It harmonizes the columns and index name of a pandas dataframe.

         Args:
            df(pd.DataFrame): data series
        """
        try:
            df.columns = self.vars2
        except ValueError:
            df.columns = self.vars2[1::]
            df['O3'] = np.nan  # for those which has not o3 data
        df.index.rename('date-time', inplace=True)

    def concat_data(self, df=None):
        """"
        It concatenates the data series.

         Args:
            df(pd.DataFrame): data series
        """
        try:
            self.df = pd.concat([self.df, df], axis=0)
        except KeyError:
            self.df = pd.concat([self.df, df], axis=0)

    def save_harmonized_data(self):
        """
        It saves the harmonized data.
        """
        self.df.sort_index().to_csv(self.path + '/' + str(self.c_year) + '.csv')

    def run(self):
        """
        It reads the reanalysis and observation data and plots them.
        """
        #df = pd.read_excel(sys.path[2] + '/../Data/doe-test-device/5.xls', header=6, index_col=[0,1,2,3])
        #df.to_csv(sys.path[2] + '/../Data/doe-test-device/5.csv')
        from datetime import datetime
        dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        import matplotlib.pyplot as plt
        df= pd.read_csv(sys.path[2] + '/../Data/doe-test-device/1.csv',
                        header=0, parse_dates={'datetime':['year','month','day','hour']},
                        index_col='datetime')
        ds = df['O3 (ppb)'].iloc[::-1]
        print(ds.autocorr(lag=3)) #corrwith(df['O3 (ppb)']))
        #ds.plot.hist(bins=12, alpha=0.5)
        #ds.plot.kde()
        #plt.show()
        #df['datetime'] = pd.to_datetime(df[["year", "month", "day"]],)
        #df['datetime'] = str(df['year'])  + '-' + str(df['month'])
        df['datehour'] = pd.to_datetime(df.index, format='%Y %m %d',errors='ignore')
        #df['n'] = df['datehour'].dt.strftime('%m/%d/%Y %H')
        #df.style.format({"datehour": lambda t: t.strftime("%d-%m-%Y %H")})
        df['datehour'] = pd.to_datetime(df['datehour'].astype(str), format='%Y/%m/%d')
        # self.convert_jalali_to_gregorian(df=df)
        idx_f = pd.Series.first_valid_index(df)
        self.convert_jalali_to_gregorian(df=df)
        print(idx_f)
        #self.correct_time(df=df)
        exit()
        self.c_year = self.f_year
        while self.c_year < int(self.l_year+1):
            self.df = pd.DataFrame()  # data series
            self.path = sys.path[0] + '/Data/PollutantTehran/Geophysic/y' + str(self.c_year)
            self.read_check_raw_data()
            self.save_harmonized_data()
            print('It is done for the year ' + str(self.c_year))
            self.c_year+=1

if __name__ == '__main__':
    HarmonizeDataSeries(vars1=['O3T008', 'NOT008', 'N2T008', 'NXT008', 'COT008', 'S2T008'],
                        vars2=['O3', 'NO', 'NO2', 'NOX', 'CO', 'SO2'],
                        vars3=['TEU103\nNO TE Urban1 3', 'TEU103\nNO2 TE Urban1 3','TEU103\nNOX TE Urban1 3','TEU103\nCO TE Urban1 3','TEU103\nSO2 TE Urban1 3'],
                        s_year=2007, e_year=2021).run()