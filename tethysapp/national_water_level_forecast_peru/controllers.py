import io
import os
import sys
import time
import json
import math
import pytz
import requests
import geoglows
import numpy as np
import pandas as pd
import datetime as dt
import hydrostats as hs
import scipy.stats as sp
from bs4 import BeautifulSoup
import hydrostats.data as hd
import plotly.graph_objs as go

from csv import writer as csv_writer
from HydroErr.HydroErr import metric_names, metric_abbr
from hs_restclient import HydroShare, HydroShareAuthBasic

from tethys_sdk.gizmos import *
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .app import NationalWaterLevelForecastPeru as app

def home(request):
    """
    Controller for the app home page.
    """

    # List of Metrics to include in context
    metric_loop_list = list(zip(metric_names, metric_abbr))

    # Retrieve a geoserver engine and geoserver credentials.
    geoserver_engine = app.get_spatial_dataset_service(
        name='main_geoserver', as_engine=True)

    geos_username = geoserver_engine.username
    geos_password = geoserver_engine.password
    my_geoserver = geoserver_engine.endpoint.replace('rest', '')

    geoserver_base_url = my_geoserver
    geoserver_workspace = app.get_custom_setting('workspace')
    region = app.get_custom_setting('region')
    geoserver_endpoint = TextInput(display_text='',
                                   initial=json.dumps([geoserver_base_url, geoserver_workspace, region]),
                                   name='geoserver_endpoint',
                                   disabled=True)

    # Available Forecast Dates
    res = requests.get('https://geoglows.ecmwf.int/api/AvailableDates/?region=central_america-geoglows', verify=False)
    data = res.json()
    dates_array = (data.get('available_dates'))

    dates = []

    for date in dates_array:
        if len(date) == 10:
            date_mod = date + '000'
            date_f = dt.datetime.strptime(date_mod, '%Y%m%d.%H%M').strftime('%Y-%m-%d %H:%M')
        else:
            date_f = dt.datetime.strptime(date, '%Y%m%d.%H%M').strftime('%Y-%m-%d')
            date = date[:-3]
        dates.append([date_f, date])
        dates = sorted(dates)

    dates.append(['Select Date', dates[-1][1]])
    dates.reverse()

    # Date Picker Options
    date_picker = DatePicker(name='datesSelect',
                             display_text='Date',
                             autoclose=True,
                             format='yyyy-mm-dd',
                             start_date=dates[-1][0],
                             end_date=dates[1][0],
                             start_view='month',
                             today_button=True,
                             initial='')

    region_index = json.load(open(os.path.join(os.path.dirname(__file__), 'public', 'geojson', 'index.json')))
    regions = SelectInput(
        display_text='Zoom to a Region:',
        name='regions',
        multiple=False,
        original=True,
        options=[(region_index[opt]['name'], opt) for opt in region_index]
    )

    context = {
        "metric_loop_list": metric_loop_list,
        "geoserver_endpoint": geoserver_endpoint,
        "date_picker": date_picker,
        "regions": regions
    }

    return render(request, 'national_water_level_forecast_peru/home.html', context)

def get_popup_response(request):
    """
    get station attributes
    """

    start_time = time.time()

    observed_data_path_file = os.path.join(app.get_app_workspace().path, 'observed_data.json')
    simulated_data_path_file = os.path.join(app.get_app_workspace().path, 'simulated_data.json')
    corrected_data_path_file = os.path.join(app.get_app_workspace().path, 'corrected_data.json')
    observed_adjusted_path_file = os.path.join(app.get_app_workspace().path, 'observed_adjusted.json')

    f = open(observed_data_path_file, 'w')
    f.close()
    f2 = open(simulated_data_path_file, 'w')
    f2.close()
    f3 = open(corrected_data_path_file, 'w')
    f3.close()
    f4 = open(observed_adjusted_path_file, 'w')
    f4.close()

    return_obj = {}

    try:
        get_data = request.GET
        # get station attributes
        watershed = get_data['watershed']
        subbasin = get_data['subbasin']
        comid = get_data['streamcomid']
        codEstacion = get_data['stationcode']
        oldCodEstacion = get_data['oldstationcode']
        nomEstacion = get_data['stationname']

        '''Get Observed Data'''
        auth = HydroShareAuthBasic(username=app.get_custom_setting('username'), password=app.get_custom_setting('password'))
        hs = HydroShare(auth=auth)
        resource_id = app.get_custom_setting('hydroshare_resource_id')
        hs.setAccessRules(resource_id, public=True)

        url = 'https://www.hydroshare.org/resource/{0}/data/contents/Water_Level_Data/{1}.csv'.format(resource_id, codEstacion)
        s = requests.get(url, verify=False).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')), index_col=0)
        df.index = pd.to_datetime(df.index)

        datesWaterLevel = df.index.tolist()
        dataWaterLevel = df.iloc[:, 0].values
        dataWaterLevel.tolist()

        if isinstance(dataWaterLevel[0], str):
            dataWaterLevel = map(float, dataWaterLevel)

        observed_df = pd.DataFrame(data=dataWaterLevel, index=datesWaterLevel, columns=['Observed Water Level'])

        hs.setAccessRules(resource_id, public=False)

        observed_data_file_path = os.path.join(app.get_app_workspace().path, 'observed_data.json')
        observed_df.reset_index(level=0, inplace=True)
        observed_df['index'] = observed_df['index'].dt.strftime('%Y-%m-%d')
        observed_df.set_index('index', inplace=True)
        observed_df.index = pd.to_datetime(observed_df.index)
        observed_df.index.name = 'datetime'
        observed_df.to_json(observed_data_file_path, orient='columns')

        min_value = observed_df['Observed Water Level'].min()

        if min_value >= 0:
            min_value = 0

        observed_adjusted = observed_df - min_value

        observed_adjusted_file_path = os.path.join(app.get_app_workspace().path, 'observed_adjusted.json')
        observed_adjusted.reset_index(level=0, inplace=True)
        observed_adjusted['datetime'] = observed_adjusted['datetime'].dt.strftime('%Y-%m-%d')
        observed_adjusted.set_index('datetime', inplace=True)
        observed_adjusted.index = pd.to_datetime(observed_adjusted.index)
        # observed_adjusted.index.name = 'datetime'
        observed_adjusted.to_json(observed_adjusted_file_path, orient='columns')

        '''Get Simulated Data'''
        simulated_df = geoglows.streamflow.historic_simulation(comid, forcing='era_5', return_format='csv')
        # Removing Negative Values
        simulated_df[simulated_df < 0] = 0
        simulated_df.index = pd.to_datetime(simulated_df.index)
        simulated_df.index = simulated_df.index.to_series().dt.strftime("%Y-%m-%d")
        simulated_df.index = pd.to_datetime(simulated_df.index)
        simulated_df = pd.DataFrame(data=simulated_df.iloc[:, 0].values, index=simulated_df.index,
                                    columns=['Simulated Streamflow'])

        simulated_data_file_path = os.path.join(app.get_app_workspace().path, 'simulated_data.json')
        simulated_df.reset_index(level=0, inplace=True)
        simulated_df['datetime'] = simulated_df['datetime'].dt.strftime('%Y-%m-%d')
        simulated_df.set_index('datetime', inplace=True)
        simulated_df.index = pd.to_datetime(simulated_df.index)
        simulated_df.index.name = 'Datetime'
        simulated_df.to_json(simulated_data_file_path)

        print("finished get_popup_response")

        print("--- %s seconds getpopup ---" % (time.time() - start_time))

        return JsonResponse({})

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))
        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

def get_hydrographs(request):
    """
    Get observed data from csv files in Hydroshare
    Get historic simulations from ERA Interim
    """

    start_time = time.time()

    try:

        get_data = request.GET
        watershed = get_data['watershed']
        subbasin = get_data['subbasin']
        comid = get_data['streamcomid']
        codEstacion = get_data['stationcode']
        oldCodEstacion = get_data['oldstationcode']
        nomEstacion = get_data['stationname']

        '''Get Observed Data'''
        observed_data_file_path = os.path.join(app.get_app_workspace().path, 'observed_data.json')
        observed_df = pd.read_json(observed_data_file_path,convert_dates=True)
        observed_df.index = pd.to_datetime(observed_df.index, unit='ms')
        observed_df.sort_index(inplace=True, ascending=True)

        min_value = observed_df['Observed Water Level'].min()

        if min_value >= 0:
            min_value = 0

        '''Get Adjusted Data'''
        observed_adjusted_file_path = os.path.join(app.get_app_workspace().path, 'observed_adjusted.json')
        observed_adjusted = pd.read_json(observed_adjusted_file_path, convert_dates=True)
        observed_adjusted.index = pd.to_datetime(observed_adjusted.index, unit='ms')
        observed_adjusted.sort_index(inplace=True, ascending=True)

        '''Get Simulated Data'''
        simulated_data_file_path = os.path.join(app.get_app_workspace().path, 'simulated_data.json')
        simulated_df = pd.read_json(simulated_data_file_path, convert_dates=True)
        simulated_df.index = pd.to_datetime(simulated_df.index)
        simulated_df.sort_index(inplace=True, ascending=True)

        '''Correct the Bias in Sumulation'''
        corrected_df = geoglows.bias.correct_historical(simulated_df, observed_adjusted)
        corrected_df = corrected_df + min_value
        corrected_data_file_path = os.path.join(app.get_app_workspace().path, 'corrected_data.json')
        corrected_df.reset_index(level=0, inplace=True)
        corrected_df['index'] = corrected_df['index'].dt.strftime('%Y-%m-%d')
        corrected_df.set_index('index', inplace=True)
        corrected_df.index = pd.to_datetime(corrected_df.index)
        corrected_df.index.name = 'Datetime'
        corrected_df.to_json(corrected_data_file_path)

        '''Plotting Data'''
        observed_WL = go.Scatter(x=observed_df.index, y=observed_df.iloc[:, 0].values, name='Observed', line=dict(color="#636EFA"))
        corrected_WL = go.Scatter(x=corrected_df.index, y=corrected_df.iloc[:, 0].values, name='Corrected Simulated', line=dict(color="#00CC96"))

        layout = go.Layout(
            title='Observed & Simulated Water Level at <br> {0} - {1}'.format(codEstacion, nomEstacion),
            xaxis=dict(title='Dates', ), yaxis=dict(title='Water Level (m)', autorange=True),
            showlegend=True)

        chart_obj = PlotlyView(go.Figure(data=[observed_WL, corrected_WL], layout=layout))

        context = {
            'gizmo_object': chart_obj,
        }

        print("--- %s seconds hydrographs ---" % (time.time() - start_time))

        return render(request, 'national_water_level_forecast_peru/gizmo_ajax.html', context)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))
        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

def get_dailyAverages(request):
    """
    Get observed data from csv files in Hydroshare
    Get historic simulations from ERA Interim
    """

    start_time = time.time()

    try:
        get_data = request.GET
        watershed = get_data['watershed']
        subbasin = get_data['subbasin']
        comid = get_data['streamcomid']
        codEstacion = get_data['stationcode']
        oldCodEstacion = get_data['oldstationcode']
        nomEstacion = get_data['stationname']

        '''Get Observed Data'''
        observed_data_file_path = os.path.join(app.get_app_workspace().path, 'observed_data.json')
        observed_df = pd.read_json(observed_data_file_path,convert_dates=True)
        observed_df.index = pd.to_datetime(observed_df.index, unit='ms')
        observed_df.sort_index(inplace=True, ascending=True)

        '''Get Bias Corrected Data'''
        corrected_data_file_path = os.path.join(app.get_app_workspace().path, 'corrected_data.json')
        corrected_df = pd.read_json(corrected_data_file_path,convert_dates=True)
        corrected_df.index = pd.to_datetime(corrected_df.index)
        corrected_df.sort_index(inplace=True, ascending=True)

        '''Merge Data'''

        merged_df2 = hd.merge_data(sim_df=corrected_df, obs_df=observed_df)

        '''Plotting Data'''

        daily_avg2 = hd.daily_average(merged_df2)

        daily_avg_obs_WL = go.Scatter(x=daily_avg2.index, y=daily_avg2.iloc[:, 1].values, name='Observed', line=dict(color="#636EFA"))

        daily_avg_corr_sim_WL = go.Scatter(x=daily_avg2.index, y=daily_avg2.iloc[:, 0].values,
                                          name='Corrected Simulated', line=dict(color="#00CC96"))

        layout = go.Layout(
            title='Daily Average Water Level for <br> {0} - {1}'.format(codEstacion, nomEstacion),
            xaxis=dict(title='Days', ), yaxis=dict(title='Water Level (m)', autorange=True),
            showlegend=True)

        chart_obj = PlotlyView(go.Figure(data=[daily_avg_obs_WL, daily_avg_corr_sim_WL], layout=layout))

        context = {
            'gizmo_object': chart_obj,
        }

        print("--- %s seconds dailyAverages ---" % (time.time() - start_time))

        return render(request, 'national_water_level_forecast_peru/gizmo_ajax.html', context)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))
        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

def get_monthlyAverages(request):
    """
    Get observed data from csv files in Hydroshare
    Get historic simulations from ERA Interim
    """

    start_time = time.time()

    try:
        get_data = request.GET
        watershed = get_data['watershed']
        subbasin = get_data['subbasin']
        comid = get_data['streamcomid']
        codEstacion = get_data['stationcode']
        oldCodEstacion = get_data['oldstationcode']
        nomEstacion = get_data['stationname']

        '''Get Observed Data'''
        observed_data_file_path = os.path.join(app.get_app_workspace().path, 'observed_data.json')
        observed_df = pd.read_json(observed_data_file_path,convert_dates=True)
        observed_df.index = pd.to_datetime(observed_df.index, unit='ms')
        observed_df.sort_index(inplace=True, ascending=True)

        '''Get Bias Corrected Data'''
        corrected_data_file_path = os.path.join(app.get_app_workspace().path, 'corrected_data.json')
        corrected_df = pd.read_json(corrected_data_file_path,convert_dates=True)
        corrected_df.index = pd.to_datetime(corrected_df.index)
        corrected_df.sort_index(inplace=True, ascending=True)

        '''Merge Data'''

        merged_df2 = hd.merge_data(sim_df=corrected_df, obs_df=observed_df)

        '''Plotting Data'''

        monthly_avg2 = hd.monthly_average(merged_df2)

        monthly_avg_obs_WL = go.Scatter(x=monthly_avg2.index, y=monthly_avg2.iloc[:, 1].values, name='Observed', line=dict(color="#636EFA"))

        monthly_avg_corr_sim_WL = go.Scatter(x=monthly_avg2.index, y=monthly_avg2.iloc[:, 0].values,
                                            name='Corrected Simulated', line=dict(color="#00CC96"))

        layout = go.Layout(
            title='Monthly Average Water Level for <br> {0} - {1}'.format(codEstacion, nomEstacion),
            xaxis=dict(title='Months', ), yaxis=dict(title='Water Level (m)', autorange=True),
            showlegend=True)

        chart_obj = PlotlyView(
            go.Figure(data=[monthly_avg_obs_WL, monthly_avg_corr_sim_WL], layout=layout))

        context = {
            'gizmo_object': chart_obj,
        }

        print("--- %s seconds monthlyAverages ---" % (time.time() - start_time))

        return render(request, 'national_water_level_forecast_peru/gizmo_ajax.html', context)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))
        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

def get_scatterPlot(request):
    """
    Get observed data from csv files in Hydroshare
    Get historic simulations from ERA Interim
    """

    start_time = time.time()

    try:
        get_data = request.GET
        watershed = get_data['watershed']
        subbasin = get_data['subbasin']
        comid = get_data['streamcomid']
        codEstacion = get_data['stationcode']
        oldCodEstacion = get_data['oldstationcode']
        nomEstacion = get_data['stationname']

        '''Get Observed Data'''
        observed_data_file_path = os.path.join(app.get_app_workspace().path, 'observed_data.json')
        observed_df = pd.read_json(observed_data_file_path,convert_dates=True)
        observed_df.index = pd.to_datetime(observed_df.index, unit='ms')
        observed_df.sort_index(inplace=True, ascending=True)

        '''Get Bias Corrected Data'''
        corrected_data_file_path = os.path.join(app.get_app_workspace().path, 'corrected_data.json')
        corrected_df = pd.read_json(corrected_data_file_path,convert_dates=True)
        corrected_df.index = pd.to_datetime(corrected_df.index)
        corrected_df.sort_index(inplace=True, ascending=True)

        '''Merge Data'''

        merged_df2 = hd.merge_data(sim_df=corrected_df, obs_df=observed_df)

        '''Plotting Data'''

        scatter_data2 = go.Scatter(
            x=merged_df2.iloc[:, 0].values,
            y=merged_df2.iloc[:, 1].values,
            mode='markers',
            name='corrected',
            marker=dict(color='#00cc96')
        )

        min_value2 = min(min(merged_df2.iloc[:, 1].values), min(merged_df2.iloc[:, 0].values))
        max_value2 = max(max(merged_df2.iloc[:, 1].values), max(merged_df2.iloc[:, 0].values))

        line_45 = go.Scatter(
            x=[min_value2, max_value2],
            y=[min_value2, max_value2],
            mode='lines',
            name='45deg line',
            line=dict(color='black')
        )

        slope2, intercept2, r_value2, p_value2, std_err2 = sp.linregress(merged_df2.iloc[:, 0].values,
                                                                         merged_df2.iloc[:, 1].values)

        line_adjusted2 = go.Scatter(
            x=[min_value2, max_value2],
            y=[slope2 * min_value2 + intercept2, slope2 * max_value2 + intercept2],
            mode='lines',
            name='{0}x + {1}'.format(str(round(slope2, 2)), str(round(intercept2, 2))),
            line=dict(color='green')
        )

        layout = go.Layout(title="Scatter Plot for {0} - {1}".format(codEstacion, nomEstacion),
                           xaxis=dict(title='Simulated', ), yaxis=dict(title='Observed', autorange=True),
                           showlegend=True)

        chart_obj = PlotlyView(
            go.Figure(data=[scatter_data2, line_45, line_adjusted2], layout=layout))

        context = {
            'gizmo_object': chart_obj,
        }

        print("--- %s seconds scatterPlot ---" % (time.time() - start_time))

        return render(request, 'national_water_level_forecast_peru/gizmo_ajax.html', context)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))
        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

def get_scatterPlotLogScale(request):
    """
    Get observed data from csv files in Hydroshare
    Get historic simulations from ERA Interim
    """

    start_time = time.time()

    try:
        get_data = request.GET
        watershed = get_data['watershed']
        subbasin = get_data['subbasin']
        comid = get_data['streamcomid']
        codEstacion = get_data['stationcode']
        oldCodEstacion = get_data['oldstationcode']
        nomEstacion = get_data['stationname']

        '''Get Observed Data'''
        observed_data_file_path = os.path.join(app.get_app_workspace().path, 'observed_data.json')
        observed_df = pd.read_json(observed_data_file_path,convert_dates=True)
        observed_df.index = pd.to_datetime(observed_df.index, unit='ms')
        observed_df.sort_index(inplace=True, ascending=True)

        '''Get Bias Corrected Data'''
        corrected_data_file_path = os.path.join(app.get_app_workspace().path, 'corrected_data.json')
        corrected_df = pd.read_json(corrected_data_file_path,convert_dates=True)
        corrected_df.index = pd.to_datetime(corrected_df.index)
        corrected_df.sort_index(inplace=True, ascending=True)

        '''Merge Data'''

        merged_df2 = hd.merge_data(sim_df=corrected_df, obs_df=observed_df)

        '''Plotting Data'''

        scatter_data2 = go.Scatter(
            x=merged_df2.iloc[:, 0].values,
            y=merged_df2.iloc[:, 1].values,
            mode='markers',
            name='corrected',
            marker=dict(color='#00cc96')
        )

        min_value = min(min(merged_df2.iloc[:, 1].values), min(merged_df2.iloc[:, 0].values))
        max_value = max(max(merged_df2.iloc[:, 1].values), max(merged_df2.iloc[:, 0].values))

        line_45 = go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode='lines',
            name='45deg line',
            line=dict(color='black')
        )

        layout = go.Layout(title="Scatter Plot for {0} - {1} (Log Scale)".format(codEstacion, nomEstacion),
                           xaxis=dict(title='Simulated', type='log', ), yaxis=dict(title='Observed', type='log',
                                                                                   autorange=True), showlegend=True)

        chart_obj = PlotlyView(go.Figure(data=[scatter_data2, line_45], layout=layout))

        context = {
            'gizmo_object': chart_obj,
        }

        print("--- %s seconds scatterPlot_log ---" % (time.time() - start_time))

        return render(request, 'national_water_level_forecast_peru/gizmo_ajax.html', context)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))
        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

def make_table_ajax(request):

    start_time = time.time()

    try:
        get_data = request.GET
        watershed = get_data['watershed']
        subbasin = get_data['subbasin']
        comid = get_data['streamcomid']
        codEstacion = get_data['stationcode']
        oldCodEstacion = get_data['oldstationcode']
        nomEstacion = get_data['stationname']

        # Indexing the metrics to get the abbreviations
        selected_metric_abbr = get_data.getlist("metrics[]", None)

        # Retrive additional parameters if they exist
        # Retrieving the extra optional parameters
        extra_param_dict = {}

        if request.GET.get('mase_m', None) is not None:
            mase_m = float(request.GET.get('mase_m', None))
            extra_param_dict['mase_m'] = mase_m
        else:
            mase_m = 1
            extra_param_dict['mase_m'] = mase_m

        if request.GET.get('dmod_j', None) is not None:
            dmod_j = float(request.GET.get('dmod_j', None))
            extra_param_dict['dmod_j'] = dmod_j
        else:
            dmod_j = 1
            extra_param_dict['dmod_j'] = dmod_j

        if request.GET.get('nse_mod_j', None) is not None:
            nse_mod_j = float(request.GET.get('nse_mod_j', None))
            extra_param_dict['nse_mod_j'] = nse_mod_j
        else:
            nse_mod_j = 1
            extra_param_dict['nse_mod_j'] = nse_mod_j

        if request.GET.get('h6_k_MHE', None) is not None:
            h6_mhe_k = float(request.GET.get('h6_k_MHE', None))
            extra_param_dict['h6_mhe_k'] = h6_mhe_k
        else:
            h6_mhe_k = 1
            extra_param_dict['h6_mhe_k'] = h6_mhe_k

        if request.GET.get('h6_k_AHE', None) is not None:
            h6_ahe_k = float(request.GET.get('h6_k_AHE', None))
            extra_param_dict['h6_ahe_k'] = h6_ahe_k
        else:
            h6_ahe_k = 1
            extra_param_dict['h6_ahe_k'] = h6_ahe_k

        if request.GET.get('h6_k_RMSHE', None) is not None:
            h6_rmshe_k = float(request.GET.get('h6_k_RMSHE', None))
            extra_param_dict['h6_rmshe_k'] = h6_rmshe_k
        else:
            h6_rmshe_k = 1
            extra_param_dict['h6_rmshe_k'] = h6_rmshe_k

        if float(request.GET.get('lm_x_bar', None)) != 1:
            lm_x_bar_p = float(request.GET.get('lm_x_bar', None))
            extra_param_dict['lm_x_bar_p'] = lm_x_bar_p
        else:
            lm_x_bar_p = None
            extra_param_dict['lm_x_bar_p'] = lm_x_bar_p

        if float(request.GET.get('d1_p_x_bar', None)) != 1:
            d1_p_x_bar_p = float(request.GET.get('d1_p_x_bar', None))
            extra_param_dict['d1_p_x_bar_p'] = d1_p_x_bar_p
        else:
            d1_p_x_bar_p = None
            extra_param_dict['d1_p_x_bar_p'] = d1_p_x_bar_p

        '''Get Observed Data'''
        observed_data_file_path = os.path.join(app.get_app_workspace().path, 'observed_data.json')
        observed_df = pd.read_json(observed_data_file_path,convert_dates=True)
        observed_df.index = pd.to_datetime(observed_df.index, unit='ms')
        observed_df.sort_index(inplace=True, ascending=True)

        '''Get Bias Corrected Data'''
        corrected_data_file_path = os.path.join(app.get_app_workspace().path, 'corrected_data.json')
        corrected_df = pd.read_json(corrected_data_file_path,convert_dates=True)
        corrected_df.index = pd.to_datetime(corrected_df.index)
        corrected_df.sort_index(inplace=True, ascending=True)

        '''Merge Data'''
        merged_df2 = hd.merge_data(sim_df=corrected_df, obs_df=observed_df)

        '''Plotting Data'''

        # Creating the Table Based on User Input
        table2 = hs.make_table(
            merged_dataframe = merged_df2,
            metrics= selected_metric_abbr,
            # remove_neg=remove_neg,
            # remove_zero=remove_zero,
            mase_m= extra_param_dict['mase_m'],
            dmod_j= extra_param_dict['dmod_j'],
            nse_mod_j= extra_param_dict['nse_mod_j'],
            h6_mhe_k= extra_param_dict['h6_mhe_k'],
            h6_ahe_k= extra_param_dict['h6_ahe_k'],
            h6_rmshe_k= extra_param_dict['h6_rmshe_k'],
            d1_p_obs_bar_p= extra_param_dict['d1_p_x_bar_p'],
            lm_x_obs_bar_p= extra_param_dict['lm_x_bar_p'],
            # seasonal_periods=all_date_range_list
        )
        table2 = table2.round(decimals=2)
        table_html2 = table2.transpose()
        table_html2 = table_html2.to_html(classes="table table-hover table-striped").replace('border="1"', 'border="0"')

        print("--- %s seconds metrics_table ---" % (time.time() - start_time))

        return HttpResponse(table_html2)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))
        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })


def get_time_series_bc(request):

    start_time = time.time()

    try:

        get_data = request.GET
        watershed = get_data['watershed']
        subbasin = get_data['subbasin']
        comid = get_data['streamcomid']
        codEstacion = get_data['stationcode']
        oldCodEstacion = get_data['oldstationcode']
        catEstacion = get_data['stationcat']
        estadoEstacion = get_data['stationstatus']
        nomEstacion = get_data['stationname']
        startdate = get_data['startdate']

        '''Get Observed Data'''
        observed_data_file_path = os.path.join(app.get_app_workspace().path, 'observed_data.json')
        observed_df = pd.read_json(observed_data_file_path, convert_dates=True)
        observed_df.index = pd.to_datetime(observed_df.index, unit='ms')
        observed_df.sort_index(inplace=True, ascending=True)

        min_value = observed_df['Observed Water Level'].min()

        if min_value >= 0:
            min_value = 0

        '''Get Adjusted Data'''
        observed_adjusted_file_path = os.path.join(app.get_app_workspace().path, 'observed_adjusted.json')
        observed_adjusted = pd.read_json(observed_adjusted_file_path, convert_dates=True)
        observed_adjusted.index = pd.to_datetime(observed_adjusted.index, unit='ms')
        observed_adjusted.sort_index(inplace=True, ascending=True)

        '''Get Simulated Data'''
        simulated_data_file_path = os.path.join(app.get_app_workspace().path, 'simulated_data.json')
        simulated_df = pd.read_json(simulated_data_file_path, convert_dates=True)
        simulated_df.index = pd.to_datetime(simulated_df.index)
        simulated_df.sort_index(inplace=True, ascending=True)

        '''Get Bias Corrected Data'''
        corrected_data_file_path = os.path.join(app.get_app_workspace().path, 'corrected_data.json')
        corrected_df = pd.read_json(corrected_data_file_path, convert_dates=True)
        corrected_df.index = pd.to_datetime(corrected_df.index)
        corrected_df.sort_index(inplace=True, ascending=True)

        '''Getting Forecast Stats'''
        if startdate != '':
            res = requests.get('https://geoglows.ecmwf.int/api/ForecastEnsembles/?reach_id=' + comid + '&date=' + startdate + '&return_format=csv', verify=False).content
        else:
            res = requests.get('https://geoglows.ecmwf.int/api/ForecastEnsembles/?reach_id=' + comid + '&return_format=csv', verify=False).content

        '''Get Forecasts'''
        forecast_ens = pd.read_csv(io.StringIO(res.decode('utf-8')), index_col=0)
        forecast_ens.index = pd.to_datetime(forecast_ens.index)
        forecast_ens[forecast_ens < 0] = 0
        forecast_ens.index = forecast_ens.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
        forecast_ens.index = pd.to_datetime(forecast_ens.index)

        forecast_ens_file_path = os.path.join(app.get_app_workspace().path, 'forecast_ens.json')
        forecast_ens.index.name = 'Datetime'
        forecast_ens.to_json(forecast_ens_file_path)

        '''Get Forecasts Records'''
        forecast_record = geoglows.streamflow.forecast_records(comid)
        forecast_record[forecast_record < 0] = 0
        forecast_record.index = forecast_record.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
        forecast_record.index = pd.to_datetime(forecast_record.index)

        '''Correct Bias Forecasts'''

        monthly_simulated = simulated_df[simulated_df.index.month == (forecast_ens.index[0]).month].dropna()
        monthly_observed = observed_df[observed_df.index.month == (forecast_ens.index[0]).month].dropna()

        min_simulated = np.min(monthly_simulated.iloc[:, 0].to_list())
        max_simulated = np.max(monthly_simulated.iloc[:, 0].to_list())

        min_factor_df = forecast_ens.copy()
        max_factor_df = forecast_ens.copy()
        forecast_ens_df = forecast_ens.copy()

        for column in forecast_ens.columns:
            tmp = forecast_ens[column].dropna().to_frame()
            min_factor = tmp.copy()
            max_factor = tmp.copy()
            min_factor.loc[min_factor[column] >= min_simulated, column] = 1
            min_index_value = min_factor[min_factor[column] != 1].index.tolist()

            for element in min_index_value:
                min_factor[column].loc[min_factor.index == element] = tmp[column].loc[tmp.index == element] / min_simulated

            max_factor.loc[max_factor[column] <= max_simulated, column] = 1
            max_index_value = max_factor[max_factor[column] != 1].index.tolist()

            for element in max_index_value:
                max_factor[column].loc[max_factor.index == element] = tmp[column].loc[tmp.index == element] / max_simulated

            tmp.loc[tmp[column] <= min_simulated, column] = min_simulated
            tmp.loc[tmp[column] >= max_simulated, column] = max_simulated

            forecast_ens_df.update(pd.DataFrame(tmp[column].values, index=tmp.index, columns=[column]))
            min_factor_df.update(pd.DataFrame(min_factor[column].values, index=min_factor.index, columns=[column]))
            max_factor_df.update(pd.DataFrame(max_factor[column].values, index=max_factor.index, columns=[column]))

        corrected_ensembles = geoglows.bias.correct_forecast(forecast_ens_df, simulated_df, observed_adjusted)
        corrected_ensembles = corrected_ensembles.multiply(min_factor_df, axis=0)
        corrected_ensembles = corrected_ensembles.multiply(max_factor_df, axis=0)
        corrected_ensembles = corrected_ensembles + min_value

        forecast_ens_bc_file_path = os.path.join(app.get_app_workspace().path, 'forecast_ens_bc.json')
        corrected_ensembles.index.name = 'Datetime'
        corrected_ensembles.to_json(forecast_ens_bc_file_path)

        ensemble = corrected_ensembles.copy()
        high_res_df = ensemble['ensemble_52_m^3/s'].to_frame()
        ensemble.drop(columns=['ensemble_52_m^3/s'], inplace=True)
        ensemble.dropna(inplace=True)
        high_res_df.dropna(inplace=True)

        max_df = ensemble.quantile(1.0, axis=1).to_frame()
        max_df.rename(columns={1.0: 'flow_max_m^3/s'}, inplace=True)

        p75_df = ensemble.quantile(0.75, axis=1).to_frame()
        p75_df.rename(columns={0.75: 'flow_75%_m^3/s'}, inplace=True)

        p25_df = ensemble.quantile(0.25, axis=1).to_frame()
        p25_df.rename(columns={0.25: 'flow_25%_m^3/s'}, inplace=True)

        min_df = ensemble.quantile(0, axis=1).to_frame()
        min_df.rename(columns={0.0: 'flow_min_m^3/s'}, inplace=True)

        mean_df = ensemble.mean(axis=1).to_frame()
        mean_df.rename(columns={0: 'flow_avg_m^3/s'}, inplace=True)

        high_res_df.rename(columns={'ensemble_52_m^3/s': 'high_res_m^3/s'}, inplace=True)

        fixed_stats = pd.concat([max_df, p75_df, mean_df, p25_df, min_df, high_res_df], axis=1)

        forecast_data_bc_file_path = os.path.join(app.get_app_workspace().path, 'forecast_data_bc.json')
        fixed_stats.index.name = 'Datetime'
        fixed_stats.to_json(forecast_data_bc_file_path)

        titles = {'Station': nomEstacion + '-' + str(codEstacion), 'Reach ID': comid, 'bias_corrected': True}

        hydroviewer_figure = geoglows.plots.forecast_stats(stats=fixed_stats, titles=titles)

        x_vals = (fixed_stats.index[0], fixed_stats.index[len(fixed_stats.index) - 1], fixed_stats.index[len(fixed_stats.index) - 1], fixed_stats.index[0])
        max_visible = max(fixed_stats.max())

        '''Correct Bias Forecast Records'''

        date_ini = forecast_record.index[0]
        month_ini = date_ini.month

        date_end = forecast_record.index[-1]
        month_end = date_end.month

        meses = np.arange(month_ini, month_end + 1, 1)

        fixed_records = pd.DataFrame()

        for mes in meses:
            values = forecast_record.loc[forecast_record.index.month == mes]

            monthly_simulated = simulated_df[simulated_df.index.month == mes].dropna()
            monthly_observed = observed_df[observed_df.index.month == mes].dropna()

            min_simulated = np.min(monthly_simulated.iloc[:, 0].to_list())
            max_simulated = np.max(monthly_simulated.iloc[:, 0].to_list())

            min_factor_records_df = values.copy()
            max_factor_records_df = values.copy()
            fixed_records_df = values.copy()

            column_records = values.columns[0]
            tmp = forecast_record[column_records].dropna().to_frame()
            min_factor = tmp.copy()
            max_factor = tmp.copy()
            min_factor.loc[min_factor[column_records] >= min_simulated, column_records] = 1
            min_index_value = min_factor[min_factor[column_records] != 1].index.tolist()

            for element in min_index_value:
                min_factor[column_records].loc[min_factor.index == element] = tmp[column_records].loc[tmp.index == element] / min_simulated

            max_factor.loc[max_factor[column_records] <= max_simulated, column_records] = 1
            max_index_value = max_factor[max_factor[column_records] != 1].index.tolist()

            for element in max_index_value:
                max_factor[column_records].loc[max_factor.index == element] = tmp[column_records].loc[tmp.index == element] / max_simulated

            tmp.loc[tmp[column_records] <= min_simulated, column_records] = min_simulated
            tmp.loc[tmp[column_records] >= max_simulated, column_records] = max_simulated

            fixed_records_df.update(pd.DataFrame(tmp[column_records].values, index=tmp.index, columns=[column_records]))
            min_factor_records_df.update(pd.DataFrame(min_factor[column_records].values, index=min_factor.index, columns=[column_records]))
            max_factor_records_df.update(pd.DataFrame(max_factor[column_records].values, index=max_factor.index, columns=[column_records]))

            corrected_values = geoglows.bias.correct_forecast(fixed_records_df, simulated_df, observed_adjusted)
            corrected_values = corrected_values.multiply(min_factor_records_df, axis=0)
            corrected_values = corrected_values.multiply(max_factor_records_df, axis=0)
            corrected_values = corrected_values + min_value
            fixed_records = fixed_records.append(corrected_values)

        fixed_records.sort_index(inplace=True)

        record_plot = fixed_records.copy()
        record_plot = record_plot.loc[record_plot.index >= pd.to_datetime(fixed_stats.index[0] - dt.timedelta(days=8))]
        record_plot = record_plot.loc[record_plot.index <= pd.to_datetime(fixed_stats.index[0])]

        if len(record_plot.index) > 0:
            hydroviewer_figure.add_trace(go.Scatter(
                name='1st days forecasts',
                x=record_plot.index,
                y=record_plot.iloc[:, 0].values,
                line=dict(
                    color='#FFA15A',
                )
            ))

            x_vals = (record_plot.index[0], fixed_stats.index[len(fixed_stats.index) - 1], fixed_stats.index[len(fixed_stats.index) - 1], record_plot.index[0])
            max_visible = max(record_plot.max().values[0], max_visible)

        '''Getting real time observed data'''

        try:

            tz = pytz.timezone('America/Bogota')
            hoy = dt.datetime.now(tz)

            if hoy.month < 10:
                MM = '0' + str(hoy.month)
            else:
                MM = str(hoy.month)

            YYYY = str(hoy.year)

            if estadoEstacion == 'DIFERIDO':
                url_rt = 'https://www.senamhi.gob.pe/mapas/mapa-estaciones-2/_dato_esta_tipo02.php?estaciones={0}&CBOFiltro={1}{2}&t_e=H&estado={3}&cod_old={4}&cate_esta={5}&alt=263'.format(codEstacion, YYYY, MM, estadoEstacion, oldCodEstacion, catEstacion)

                try:
                    page = requests.get(url_rt)
                    soup = BeautifulSoup(page.content, 'html.parser')

                    results = soup.find(id='dataTable')
                    df_stations = pd.read_html(str(results))[0]
                    df_stations = df_stations.loc[df_stations.index >= 2]

                    if len(df_stations.iloc[:, 0].values) > 0:
                        dates = df_stations.iloc[:, 0].values
                        values_06hrs = df_stations.iloc[:, 1].values
                        values_10hrs = df_stations.iloc[:, 2].values
                        values_14hrs = df_stations.iloc[:, 3].values
                        values_18hrs = df_stations.iloc[:, 4].values

                        FECHAS = []
                        values = []

                        for i in range(0, len(dates)):
                            FECHAS.append(dt.datetime(int(dates[i][0:4]), int(dates[i][5:7]), int(dates[i][8:10]), 6, 0, 0))
                            FECHAS.append(dt.datetime(int(dates[i][0:4]), int(dates[i][5:7]), int(dates[i][8:10]), 10, 0, 0))
                            FECHAS.append(dt.datetime(int(dates[i][0:4]), int(dates[i][5:7]), int(dates[i][8:10]), 14, 0, 0))
                            FECHAS.append(dt.datetime(int(dates[i][0:4]), int(dates[i][5:7]), int(dates[i][8:10]), 18, 0, 0))

                            if values_06hrs[i] == 'S/D':
                                values.append(np.nan)
                            else:
                                values.append(float(values_06hrs[i]))
                            if values_10hrs[i] == 'S/D':
                                values.append(np.nan)
                            else:
                                values.append(float(values_10hrs[i]))
                            if values_14hrs[i] == 'S/D':
                                values.append(np.nan)
                            else:
                                values.append(float(values_14hrs[i]))
                            if values_18hrs[i] == 'S/D':
                                values.append(np.nan)
                            else:
                                values.append(float(values_18hrs[i]))

                    else:
                        print('No hay datos para la estación')

                except Exception as e:
                    print(e)

            elif estadoEstacion == 'REAL':

                url_rt = 'https://www.senamhi.gob.pe/mapas/mapa-estaciones-2/_dato_esta_tipo02.php?estaciones={0}&CBOFiltro={1}{2}&t_e=H&estado={3}&cod_old={4}&cate_esta={5}&alt=101'.format(codEstacion, YYYY, MM, estadoEstacion, oldCodEstacion, catEstacion)

                try:
                    page = requests.get(url_rt)
                    soup = BeautifulSoup(page.content, 'html.parser')

                    results = soup.find(id='dataTable')
                    df_stations = pd.read_html(str(results))[0]
                    df_stations = df_stations.loc[df_stations.index >= 2]

                    if len(df_stations.iloc[:, 0].values) > 0:
                        dates = df_stations.iloc[:, 0].values
                        values_06hrs = df_stations.iloc[:, 1].values
                        values_10hrs = df_stations.iloc[:, 2].values
                        values_14hrs = df_stations.iloc[:, 3].values
                        values_18hrs = df_stations.iloc[:, 4].values

                        FECHAS = []
                        values = []

                        for i in range(0, len(dates)):
                            FECHAS.append(dt.datetime(int(dates[i][0:4]), int(dates[i][5:7]), int(dates[i][8:10]), 6, 0, 0))
                            FECHAS.append(dt.datetime(int(dates[i][0:4]), int(dates[i][5:7]), int(dates[i][8:10]), 10, 0, 0))
                            FECHAS.append(dt.datetime(int(dates[i][0:4]), int(dates[i][5:7]), int(dates[i][8:10]), 14, 0, 0))
                            FECHAS.append(dt.datetime(int(dates[i][0:4]), int(dates[i][5:7]), int(dates[i][8:10]), 18, 0, 0))

                            if values_06hrs[i] == 'S/D':
                                values.append(np.nan)
                            else:
                                values.append(float(values_06hrs[i]))
                            if values_10hrs[i] == 'S/D':
                                values.append(np.nan)
                            else:
                                values.append(float(values_10hrs[i]))
                            if values_14hrs[i] == 'S/D':
                                values.append(np.nan)
                            else:
                                values.append(float(values_14hrs[i]))
                            if values_18hrs[i] == 'S/D':
                                values.append(np.nan)
                            else:
                                values.append(float(values_18hrs[i]))

                    else:
                        print('No hay datos para la estación')

                except Exception as e:
                    print(e)

            elif estadoEstacion == 'AUTOMATICA':

                url_rt = 'https://www.senamhi.gob.pe/mapas/mapa-estaciones-2/_dato_esta_tipo02.php?estaciones={0}&CBOFiltro={1}{2}&t_e=H&estado={3}&cod_old={4}&cate_esta={5}&alt=280'.format(codEstacion, YYYY, MM, estadoEstacion, oldCodEstacion, catEstacion)

                try:
                    page = requests.get(url_rt)
                    soup = BeautifulSoup(page.content, 'html.parser')

                    results = soup.find(id='dataTable')
                    df_stations = pd.read_html(str(results))[0]
                    df_stations = df_stations.loc[df_stations.index >= 2]

                    if len(df_stations.iloc[:, 0].values) > 0:
                        dates = df_stations.iloc[:, 0].values
                        horas = df_stations.iloc[:, 1].values
                        niveles = df_stations.iloc[:, 2].values

                        try:
                            precipitacion = df_stations.iloc[:, 3].values
                        except IndexError:
                            print('No hay datos de lluvia en esta estación')

                        FECHAS = []
                        values = []
                        lluvia = []

                        for i in range(0, len(dates)):
                            FECHAS.append(dt.datetime(int(dates[i][0:4]), int(dates[i][5:7]), int(dates[i][8:10]),int(horas[i][0:2]), int(horas[i][3:5])))

                            if niveles[i] == 'S/D':
                                values.append(np.nan)
                            else:
                                values.append(float(niveles[i]))

                            try:
                                if precipitacion[i] == 'S/D':
                                    lluvia.append(np.nan)
                                else:
                                    lluvia.append(float(precipitacion[i]))
                            except IndexError:
                                print('No hay datos de lluvia en esta estación')

                    else:
                        print('No hay datos para la estación')

                except Exception as e:
                    print(e)

            pairs = [list(a) for a in zip(FECHAS, values)]
            observed_rt = pd.DataFrame(pairs, columns=['Datetime', 'Observed Water Level (m)'])
            observed_rt.set_index('Datetime', inplace=True)
            observed_rt = observed_rt.dropna()
            observed_rt.index = pd.to_datetime(observed_rt.index)
            observed_rt = observed_rt.dropna()

            observed_rt_plot = observed_rt.copy()
            observed_rt_plot = observed_rt_plot.loc[observed_rt_plot.index >= pd.to_datetime(forecast_ens.index[0] - dt.timedelta(days=8))]
            observed_rt_plot = observed_rt_plot.loc[observed_rt_plot.index <= pd.to_datetime(forecast_ens.index[0] + dt.timedelta(days=2))]

            if len(observed_rt_plot.index) > 0:
                hydroviewer_figure.add_trace(go.Scatter(
                    name='Observed Water Level',
                    x=observed_rt_plot.index,
                    y=observed_rt_plot.iloc[:, 0].values,
                    line=dict(
                        color='green',
                    )
                ))

                max_visible = max(observed_rt_plot.max().values[0], max_visible)

        except Exception as e:
            print(e)
            print('There is no real time data for the selected station')

        '''Getting Corrected Return Periods'''
        max_annual_flow = corrected_df.groupby(corrected_df.index.strftime("%Y")).max()
        mean_value = np.mean(max_annual_flow.iloc[:,0].values)
        std_value = np.std(max_annual_flow.iloc[:,0].values)

        return_periods = [100, 50, 25, 10, 5, 2]

        def gumbel_1(std: float, xbar: float, rp: int or float) -> float:
            """
            Solves the Gumbel Type I probability distribution function (pdf) = exp(-exp(-b)) where b is the covariate. Provide
            the standard deviation and mean of the list of annual maximum flows. Compare scipy.stats.gumbel_r
            Args:
                std (float): the standard deviation of the series
                xbar (float): the mean of the series
                rp (int or float): the return period in years
            Returns:
                float, the flow corresponding to the return period specified
            """
            # xbar = statistics.mean(year_max_flow_list)
            # std = statistics.stdev(year_max_flow_list, xbar=xbar)
            return -math.log(-math.log(1 - (1 / rp))) * std * .7797 + xbar - (.45 * std)

        return_periods_values = []

        for rp in return_periods:
            return_periods_values.append(gumbel_1(std_value, mean_value, rp))

        d = {'rivid': [comid], 'return_period_100': [return_periods_values[0]], 'return_period_50': [return_periods_values[1]], 'return_period_25': [return_periods_values[2]], 'return_period_10': [return_periods_values[3]], 'return_period_5': [return_periods_values[4]], 'return_period_2': [return_periods_values[5]]}
        rperiods = pd.DataFrame(data=d)
        rperiods.set_index('rivid', inplace=True)

        r2 = int(rperiods.iloc[0]['return_period_2'])

        colors = {
            '2 Year': 'rgba(254, 240, 1, .4)',
            '5 Year': 'rgba(253, 154, 1, .4)',
            '10 Year': 'rgba(255, 56, 5, .4)',
            '20 Year': 'rgba(128, 0, 246, .4)',
            '25 Year': 'rgba(255, 0, 0, .4)',
            '50 Year': 'rgba(128, 0, 106, .4)',
            '100 Year': 'rgba(128, 0, 246, .4)',
        }

        if max_visible > r2:
            visible = True
            hydroviewer_figure.for_each_trace(
                lambda trace: trace.update(visible=True) if trace.name == "Maximum & Minimum Flow" else (),
            )
        else:
            visible = 'legendonly'
            hydroviewer_figure.for_each_trace(
                lambda trace: trace.update(visible=True) if trace.name == "Maximum & Minimum Flow" else (),
            )

        def template(name, y, color, fill='toself'):
            return go.Scatter(
                name=name,
                x=x_vals,
                y=y,
                legendgroup='returnperiods',
                fill=fill,
                visible=visible,
                line=dict(color=color, width=0))

        r5 = round(rperiods.iloc[0]['return_period_5'], 2)
        r10 = round(rperiods.iloc[0]['return_period_10'], 2)
        r25 = round(rperiods.iloc[0]['return_period_25'], 2)
        r50 = round(rperiods.iloc[0]['return_period_50'], 2)
        r100 = round(rperiods.iloc[0]['return_period_100'], 2)

        hydroviewer_figure.add_trace(template('Return Periods', (r100 * 0.05, r100 * 0.05, r100 * 0.05, r100 * 0.05), 'rgba(0,0,0,0)', fill='none'))
        hydroviewer_figure.add_trace(template(f'2 Year: {r2}', (r2, r2, r5, r5), colors['2 Year']))
        hydroviewer_figure.add_trace(template(f'5 Year: {r5}', (r5, r5, r10, r10), colors['5 Year']))
        hydroviewer_figure.add_trace(template(f'10 Year: {r10}', (r10, r10, r25, r25), colors['10 Year']))
        hydroviewer_figure.add_trace(template(f'25 Year: {r25}', (r25, r25, r50, r50), colors['25 Year']))
        hydroviewer_figure.add_trace(template(f'50 Year: {r50}', (r50, r50, r100, r100), colors['50 Year']))
        hydroviewer_figure.add_trace(template(f'100 Year: {r100}', (r100, r100, max(r100 + r100 * 0.05, max_visible), max(r100 + r100 * 0.05, max_visible)), colors['100 Year']))

        # PLOTTING AUXILIARY FUNCTIONS
        def _build_title(base, title_headers):
            if not title_headers:
                return base
            if 'bias_corrected' in title_headers.keys():
                base = 'Bias Corrected ' + base
            for head in title_headers:
                if head == 'bias_corrected':
                    continue
                base += f'<br>{head}: {title_headers[head]}'
            return base

        hydroviewer_figure['layout']['yaxis'].update(title='Water Level (m)')
        hydroviewer_figure['layout'].update(title=_build_title('Forecasted Water Level', titles))

        chart_obj = PlotlyView(hydroviewer_figure)

        context = {
            'gizmo_object': chart_obj,
        }

        print("--- %s seconds forecasts_bc ---" % (time.time() - start_time))

        return render(request, 'national_water_level_forecast_peru/gizmo_ajax.html', context)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))
        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

def get_available_dates(request):
    get_data = request.GET
    watershed = get_data['watershed']
    subbasin = get_data['subbasin']
    comid = get_data['streamcomid']

    res = requests.get('https://geoglows.ecmwf.int/api/AvailableDates/?region=' + watershed + '-' + subbasin, verify=False)

    data = res.json()

    dates_array = (data.get('available_dates'))

    dates = []

    for date in dates_array:
        if len(date) == 10:
            date_mod = date + '000'
            date_f = dt.datetime.strptime(date_mod, '%Y%m%d.%H%M').strftime('%Y-%m-%d %H:%M')
        else:
            date_f = dt.datetime.strptime(date, '%Y%m%d.%H%M').strftime('%Y-%m-%d')
            date = date[:-3]
        dates.append([date_f, date, watershed, subbasin, comid])

    dates.append(['Select Date', dates[-1][1]])
    dates.reverse()

    return JsonResponse({
        "success": "Data analysis complete!",
        "available_dates": json.dumps(dates)
    })

def get_observed_water_level_csv(request):
    """
    Get observed data from csv files in Hydroshare
    """

    try:
        get_data = request.GET
        watershed = get_data['watershed']
        subbasin = get_data['subbasin']
        comid = get_data['streamcomid']
        codEstacion = get_data['stationcode']
        nomEstacion = get_data['stationname']

        '''Get Observed Data'''
        observed_data_file_path = os.path.join(app.get_app_workspace().path, 'observed_data.json')
        observed_df = pd.read_json(observed_data_file_path, convert_dates=True)
        observed_df.index = pd.to_datetime(observed_df.index, unit='ms')
        observed_df.sort_index(inplace=True, ascending=True)

        datesObservedWaterLevel = observed_df.index.tolist()
        observedWaterLevel = observed_df.iloc[:, 0].values
        observedWaterLevel.tolist()

        pairs = [list(a) for a in zip(datesObservedWaterLevel, observedWaterLevel)]

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=observed_water_level_{0}.csv'.format(codEstacion)

        writer = csv_writer(response)
        writer.writerow(['Datetime', 'Water Level (m)'])

        for row_data in pairs:
            writer.writerow(row_data)

        return response

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))
        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

def get_simulated_bc_water_level_csv(request):
    """
    Get historic simulations from ERA Interim
    """

    try:

        get_data = request.GET
        watershed = get_data['watershed']
        subbasin = get_data['subbasin']
        comid = get_data['streamcomid']
        codEstacion = get_data['stationcode']
        nomEstacion = get_data['stationname']

        '''Get Bias Corrected Data'''
        corrected_data_file_path = os.path.join(app.get_app_workspace().path, 'corrected_data.json')
        corrected_df = pd.read_json(corrected_data_file_path, convert_dates=True)
        corrected_df.index = pd.to_datetime(corrected_df.index)
        corrected_df.sort_index(inplace=True, ascending=True)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=corrected_simulated_water_level_{0}.csv'.format(codEstacion)

        corrected_df.to_csv(encoding='utf-8', header=True, path_or_buf=response)

        return response

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))
        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

def get_forecast_bc_data_csv(request):
    """""
    Returns Forecast data as csv
    """""

    try:
        get_data = request.GET
        # get station attributes
        watershed = get_data['watershed']
        subbasin = get_data['subbasin']
        comid = get_data['streamcomid']
        startdate = get_data['startdate']

        '''Get Bias-Corrected Forecast Data'''
        forecast_data_bc_file_path = os.path.join(app.get_app_workspace().path, 'forecast_data_bc.json')
        fixed_stats = pd.read_json(forecast_data_bc_file_path, convert_dates=True)
        fixed_stats.index = pd.to_datetime(fixed_stats.index)
        fixed_stats.sort_index(inplace=True, ascending=True)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=corrected_water_level_forecast_{0}_{1}_{2}_{3}.csv'.format(watershed, subbasin, comid, startdate)

        fixed_stats.to_csv(encoding='utf-8', header=True, path_or_buf=response)

        return response

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))
        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

def get_forecast_ensemble_bc_data_csv(request):
    """""
    Returns Forecast data as csv
    """""

    get_data = request.GET

    try:
        # get station attributes
        watershed = get_data['watershed']
        subbasin = get_data['subbasin']
        comid = get_data['streamcomid']
        startdate = get_data['startdate']

        '''Get Forecast Ensemble Data'''
        forecast_ens_bc_file_path = os.path.join(app.get_app_workspace().path, 'forecast_ens_bc.json')
        corrected_ensembles = pd.read_json(forecast_ens_bc_file_path, convert_dates=True)
        corrected_ensembles.index = pd.to_datetime(corrected_ensembles.index)
        corrected_ensembles.sort_index(inplace=True, ascending=True)

        # Writing CSV
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=corrected_water_level_ensemble_forecast_{0}_{1}_{2}_{3}.csv'.format(watershed, subbasin, comid, startdate)

        corrected_ensembles.to_csv(encoding='utf-8', header=True, path_or_buf=response)

        return response

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))

        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

