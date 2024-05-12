from datetime import datetime
from datetime import timedelta

from mockseries.noise import RedNoise
from mockseries.utils import datetime_range
from mockseries.utils import plot_timeseries
from mockseries.seasonality import SinusoidalSeasonality
from mockseries.trend import LinearTrend

trend = LinearTrend(coefficient=2, time_unit=timedelta(hours=1), flat_base=100)
seasonality = SinusoidalSeasonality(amplitude=20, period=timedelta(hours=1)) \
              + SinusoidalSeasonality(amplitude=4, period=timedelta(hours=1))
noise = RedNoise(mean=0, std=3, correlation=0.5)

timeseries = trend + seasonality + noise

ts_index = datetime_range(
    granularity=timedelta(seconds=10),
    start_time=datetime(2001, 1, 1),
    num_points=1000
)
ts_values = timeseries.generate(ts_index)
plot_timeseries(ts_index, ts_values, save_path="plot/hello_mockseries.png")
