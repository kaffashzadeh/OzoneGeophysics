"""
This program is written to analysis and plot the ozone data time series.
"""
# imports python standard libraries
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

font = {'family':'Serif', 'weight':'normal', 'size':12} # 'DejaVu Sans'
matplotlib.rc('font', **font)

__author__='Najmeh Kaffashzadeh'
__author_email__ = 'n.kaffashzadeh@ut.ac.ir'


class Assess:

    name = 'Assess the time series'

    def __init__(self):
        """
        This initializes the variables.
        """
        self.df_obs = None

    def read_obs_data(self):
        """
        It reads the observed data form the given file.
        """
        self.df_obs = pd.read_csv(sys.path[0] + '/../data/O3_GeoPhys_spect.csv',
                                  parse_dates=True, index_col=0)

    def plot_box(self):
        """
        It plots the data frame as box plot with (or without) adjacent scatter.
        """
        for col in ['SY','ID', 'DU', 'SY', 'SE', 'BL']:
            df = self.df_obs[col].dropna()
            # to plot each year
            # years = [2007, 2008, 2009, 2011, 2019, 2020]
            # vals = [df[df.index.year==i].dropna().values.tolist() for i in years]
            # to plot each season of the year
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13., 8), facecolor='w')
            vals = []
            # for y in [2007, 2008, 2009, 2011, 2019, 2020]:
            # df_1y = df[df.index.year==y]
            resampled = df.resample('QS-DEC')
            for seas, grp in resampled:
                vals.append(grp.dropna().values.tolist())
            vals_sel = [vals[0], vals[1], vals[2], vals[3], [ ],
                        vals[4], vals[5], vals[6], vals[7],[ ],
                        vals[48], vals[49], vals[50], vals[51], [ ],
                        vals[52], vals[53], vals[54], vals[55]]
            ax.boxplot(vals_sel, showmeans=True, showfliers=False,
                       medianprops=dict(linestyle='-', linewidth=1.5, color='orange'),
                       meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick'))
            ax.set_ylabel('values', fontsize=16)
            ax.set_xlabel('seasons', fontsize=16)
            labels_seas =  ['DFJ','MAM','JJA','SON', '',
                            'DFJ','MAM','JJA','SON', '',
                            'DFJ','MAM','JJA','SON', '',
                            'DFJ','MAM','JJA','SON', '']
            labels_year = ['2007','2008','2019','2020','']
            # labels_year = ['2019', '2020']
            ax2 = ax.twiny()
            ax.set_xticks(np.arange(1, 21, 1))
            # ax.set_xticks(np.arange(1, 10, 1))
            ax.set_xticklabels(labels_seas, color='k', rotation=90, ha='right')
            ax2.set_xticks([1.5, 5.2, 9, 13, 15])
            # ax2.set_xticks([0.2, 0.8])
            ax2.set_xticklabels(labels_year, color='r')
            self.save_fig(fn=col + '_boxplot_seas')
            plt.close()
            # self.save_fig(fn=col+'_boxplot_seas')
            # plt.setp(bp['fliers'], markersize=1.0)
            # ax.scatter(np.full(shape=len(df_sel), fill_value=0.9), df_sel, color='green', marker='o', s=10)

    def cal_var(self, df=None):
        """
        It calculates the variance.

        Args:
            df(pd.DataFrame): data series
        """
        return df.var(axis=0)

    def calc_corr(self, df1=None, df2=None):
        """
        It calculates the correlation between two series.

        Args:
            df1(pd.DataFrame): the first data series
            df2(pd.DataFrame): the second data series
        """
        return df1.corr(df2)

    def cal_cov(self, df1=None, df2=None):
        """
        It calculates the covariance between two series.

        Args:
           df1(pd.DataFrame): the first data series
           df2(pd.DataFrame): the second data series
        """
        return self.calc_corr(df1=df1, df2=df2) * \
               np.sqrt(self.var(df=df1) * self.var(df=df2))

    def plot_pie(self, ax=None):
        """
        It plots the relative frequency of the variance of the series.

        Args:
            ax(object): axes
        """
        ax.pie([v for k, v in self.var.items()], labels=['ID','DU','SY','SE','BL'],
               autopct='%1.2f', colors=['yellow', 'limegreen', 'deepskyblue','royalblue','k'])
               # startangle=90, fontsize=12, \
               # autopct='%1.1f%%', label=' ', labeldistance=1.1, \
               # wedgeprops=dict(linewidth=1, edgecolor='w'))

    def plot_time_series(self, df=None, ax=None):
        """"
        It plots the time series of the dataframe.
        """
        df.plot(ax=ax)
        plt.xlabel('date-time')
        plt.ylabel('value')

    def save_fig(self, fn=None):
        """
        It save figure.

        Args:
            fn(str): file name
        """
        plt.savefig('../plots/'+fn+'.png', bbox_inches='tight')
        plt.close()

    def count_nums(self, df=None):
        """
        It counts and save the number (not nan value) of the series.

        Args:
            df(pd.DataFrame): data series
        """
        count = df.resample('Y').count()
        fig, ax= plt.subplots(ncols=1, nrows=1, facecolor='w')
        count['ORG'].plot(ax=ax, kind='bar', color='b', use_index=False),
        labels_year= np.arange(2007, 2022, 1)
        plt.ylabel('The number of the available data')
        plt.xlabel('years')
        plt.axhline(y=7008, color='r', linestyle='--')
        ax.set_xticklabels(labels_year, color='k', rotation=90, ha='right')
        plt.savefig('count_vals.png', bbox_inches='tight')
        plt.close()
        count.to_csv('../data/count.csv')

    def plot_ts_spect(self):
        """
        It plots the time series with spectral components.
        """
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18., 8.), facecolor='w')
        self.df_obs['ORG'].plot(ax=ax[0,0], c='k', title='ORG') # sharex=ax[1,0],
        self.df_obs['ID'].plot(ax=ax[0,1], c='b', title='ID')  # sharex=ax[1,1],
        self.df_obs['DU'].plot(ax=ax[0,2], c='b', title='DU') # sharex=ax[1,2]
        self.df_obs['SY'].plot(ax=ax[1,0], c='b', title='SY')
        self.df_obs['SE'].plot(ax=ax[1,1], c='b', title='SE')
        self.df_obs['BL'].plot(ax=ax[1,2], c='b', title='BL')
        plt.subplots_adjust(wspace=0.2, hspace=0.5)
        plt.savefig('ts_spect.png', bbox_inches='tight')
        plt.close()

    def plot_ts_1w(self, df=None, col=None):
        """
        It plots the time series for one week.
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13., 8), facecolor='w')
        ax2 = ax.twiny()
        df.plot(ax=ax, color='b')
        ax2.set_xticks([0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9])
        labels = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        ax2.set_xticklabels(labels, color='r')
        plt.savefig(col+'_ts_1w.png', bbox_inches='tight')
        plt.close()

    def run(self):
        """
        It reads the observation data and analyses them.
        """
        self.read_obs_data()
        # self.df_obs = pd.concat([self.df_obs['2007':'2008'], self.df_obs['2019':'2020']])
        # https://stackoverflow.com/questions/68378791/boxplot-of-weekends-vs-weekdays-data-in-python
        # self.df_obs = self.df_obs['2019']
        # self.df_obs['date'] = self.df_obs.index
        # self.df_obs["weekday"] = self.df_obs.apply(lambda x: x["date"].weekday(), axis=1)
        # self.df_obs["weekend"] = ((self.df_obs["weekday"] == 3) | (self.df_obs["weekday"]==4)).astype(int)
        # dic = {0: 'Weekday', 1: 'Weekday', 2:'Weekday',
        #        3:'Weekend', 4:'Weekend', 5:'Weekday', 6:'Weekday'}
        # self.df_obs['month'] = self.df_obs["date"].dt.strftime('%b,%Y')
        # self.df_obs['year'] = self.df_obs["date"].index.year#.strftime('%b,%Y')
        # self.df_obs['day'] = self.df_obs.weekday.map(dic)
        # print(self.df_obs['2019-08'])
        # self.df_obs['2019-08']['DU'].plot()
        # a = plt.figure(figsize=(13,5 ))
        # p = sns.boxplot('year', 'DU', hue='day', width=0.6, fliersize=3,
        #                    data=self.df_obs)
        # plt.savefig('weekdays_vs_weekend_du.png', bbox_inches='tight')
        # self.plot_ts_1w(df=self.df_obs['SE']['2019-08-03':'2019-08-09'], col='SE')
        # self.count_nums(df=self.df_obs)
        # self.plot_box()
        # exit()
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8., 8.), facecolor='w')
        for y,i,j in [(2007,0,0), (2008,0,1), (2019, 1,0), (2020,1,1)]:
            self.var = {}
            df_1y = self.df_obs[self.df_obs.index.year == y]
            # df_1y['ID'].plot()
            for col in ['ID','DU','SY','SE','BL']:
                # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13., 8.), facecolor='w')
                # self.plot_time_series(df=self.df_obs[col], ax=ax)
                # self.save_fig(fn=col)
                var_tmp = {col: self.cal_var(df=df_1y[col])}
                self.var.update(var_tmp)
            self.plot_pie(ax=ax[i,j])
            ax[i,j].set_title(y)
            # print('It is done for the year ' + str(y))
        self.save_fig(fn=str(y)+'_var_pie')

if __name__ == '__main__':
    Assess().run()