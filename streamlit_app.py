# streamlit_app.py
import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
import pandas as pd

# --- Matplotlib 한글 폰트 설정 ---
# Pretendard 폰트 강제 지정 대신 시스템의 기본 폰트를 사용합니다.
# 만약 한글이 깨진다면, 시스템에 맞는 한글 폰트를 직접 지정해야 할 수 있습니다.
from matplotlib import rcParams
import platform

if platform.system() == 'Windows':
    font_name = 'Malgun Gothic'
elif platform.system() == 'Darwin': # macOS
    font_name = 'AppleGothic'
else: # Linux
    # 시스템에 나눔고딕이 설치되어 있는지 확인 후 사용합니다.
    # 설치되어 있지 않다면 다른 폰트를 지정하거나, 기본 폰트를 사용하게 됩니다.
    try:
        import matplotlib.font_manager as fm
        if 'NanumGothic' in [f.name for f in fm.fontManager.ttflist]:
            font_name = 'NanumGothic'
        else:
            # 나눔고딕이 없을 경우 경고 메시지를 표시하고 기본값으로 둡니다.
            st.warning("나눔고딕 폰트가 설치되어 있지 않아 한글이 깨질 수 있습니다. 'sudo apt-get install fonts-nanum*' 명령어로 설치할 수 있습니다.")
            pass # font_name을 설정하지 않음
    except ImportError:
        pass # matplotlib.font_manager를 사용할 수 없는 경우

# font_name이 성공적으로 설정된 경우에만 rcParams를 변경합니다.
if 'font_name' in locals():
    rcParams['font.family'] = font_name

rcParams['axes.unicode_minus'] = False


# --- Streamlit 기본 설정 ---
st.set_page_config(layout="wide", page_title="기후 위기 데이터 대시보드: 해수면 온도")
st.title("NOAA OISST 해수면 온도 및 평년 편차 시각화")
st.markdown("데이터 소스: [NOAA PSL OISST v2 High Resolution](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html)")

# --- 연도별 데이터 소스 URL (OPeNDAP) ---
BASE_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"

# --- 데이터 로딩 함수 ---
@st.cache_data(show_spinner=False)
def load_and_slice_data(selected_date: datetime.date):
    """
    선택한 날짜(YYYY-MM-DD)의 한국/동중국해 인근(위도 28~42N, 경도 120~135E) SST를 로드.
    """
    year = selected_date.year
    data_url = BASE_URL.format(year=year)
    date_str = selected_date.strftime("%Y-%m-%d")

    try:
        try:
            ds = xr.open_dataset(data_url)
        except Exception:
            ds = xr.open_dataset(data_url, engine="pydap")

        da = (
            ds["sst"]
            .sel(time=date_str, lat=slice(28, 42), lon=slice(120, 135))
            .squeeze()
        )
        da.load()

        if hasattr(da, "values") and np.all(np.isnan(da.values)):
            return None
        return da

    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        st.info("연도별 파일만 제공됩니다. 네트워크(방화벽/SSL) 또는 엔진(pydap, netCDF4) 설치 문제일 수 있어요.")
        return None

# --- 평년값 계산 함수 ---
@st.cache_data(show_spinner=False)
def load_climatology_data(selected_date: datetime.date):
    """
    선택한 날짜(MM-DD)에 해당하는 30년(1991-2020) 평균 데이터를 계산하여 로드.
    2월 29일은 포함하지 않음.
    """
    climatology_period = range(1991, 2021)
    
    if selected_date.month == 2 and selected_date.day == 29:
        st.warning("2월 29일의 평년값은 제공되지 않습니다. 2월 28일 데이터를 기준으로 표시합니다.")
        target_day = selected_date.replace(day=28)
    else:
        target_day = selected_date
        
    daily_data_list = []
    
    for year in climatology_period:
        if not pd.to_datetime(f"{year}-01-01").is_leap_year and target_day.month == 2 and target_day.day == 29:
            continue
            
        date_in_year = target_day.replace(year=year)
        with st.spinner(f"{year}년 {target_day.month}월 {target_day.day}일 데이터 로드 중..."):
            daily_data = load_and_slice_data(date_in_year)
            if daily_data is not None:
                daily_data_list.append(daily_data)
    
    if not daily_data_list:
        return None
        
    climatology = xr.concat(daily_data_list, dim="time").mean(dim="time")
    climatology.load()
    return climatology


# --- 지도 시각화 함수 ---
def create_map_figure(data_array, selected_date):
    from matplotlib.colors import TwoSlopeNorm
    if data_array is None or getattr(data_array, "size", 0) == 0:
        return None

    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    
    vmin = np.nanpercentile(data_array.values, 5)
    vmax = np.nanpercentile(data_array.values, 95)

    im = data_array.plot.pcolormesh(
        ax=ax,
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )

    ax.coastlines()
    ax.add_feature(cfeature.LAND, zorder=1, facecolor="lightgray", edgecolor="black")
    
    gl = ax.gridlines(draw_labels=True, linewidth=1, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.05, aspect=40)
    cbar.set_label("해수면 온도 (°C)")
    ax.set_title(f"해수면 온도: {selected_date.strftime('%Y년 %m월 %d일')}", fontsize=16)

    fig.tight_layout()
    return fig

# --- 평년 편차 지도 시각화 함수 ---
def create_anomaly_map_figure(data_array, selected_date):
    from matplotlib.colors import TwoSlopeNorm
    if data_array is None or getattr(data_array, "size", 0) == 0:
        return None

    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    max_abs_val = np.nanmax(np.abs(data_array.values))
    norm = TwoSlopeNorm(vcenter=0, vmin=-max_abs_val, vmax=max_abs_val)

    im = data_array.plot.pcolormesh(
        ax=ax,
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        norm=norm,
        add_colorbar=False
    )

    ax.coastlines()
    ax.add_feature(cfeature.LAND, zorder=1, facecolor="lightgray", edgecolor="black")

    gl = ax.gridlines(draw_labels=True, linewidth=1, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.05, aspect=40)
    cbar.set_label("평년 대비 온도 편차 (°C)")
    ax.set_title(f"해수면 온도 편차: {selected_date.strftime('%Y년 %m월 %d일')}\n(1991-2020년 평균 대비)", fontsize=16)

    fig.tight_layout()
    return fig


# --- 사이드바 UI ---
st.sidebar.header("날짜 선택")
default_date = datetime.date.today() - datetime.timedelta(days=3)
selected_date = st.sidebar.date_input(
    "보고 싶은 날짜를 선택하세요",
    value=default_date,
    min_value=datetime.date(1981, 9, 1),
    max_value=default_date,
)

# --- 메인 로직 ---
if selected_date:
    with st.spinner(f"{selected_date:%Y-%m-%d} 데이터를 불러오는 중..."):
        sst_data = load_and_slice_data(selected_date)

    if sst_data is not None and sst_data.size > 0:
        with st.spinner(f"{selected_date:%m월 %d일}의 평년(1991-2020) 데이터를 계산하는 중... (최초 실행 시 몇 분 소요될 수 있습니다)"):
            climatology_data = load_climatology_data(selected_date)

        if climatology_data is not None:
            anomaly_data = sst_data - climatology_data
        else:
            anomaly_data = None

        tab1, tab2 = st.tabs(["🌡️ 오늘의 해수면 온도", "📊 평년 편차 (Anomaly)"])

        with tab1:
            st.subheader(f"{selected_date:%Y년 %m월 %d일} 해수면 온도 지도")
            fig_sst = create_map_figure(sst_data, selected_date)
            if fig_sst:
                st.pyplot(fig_sst, clear_figure=True)
            with st.expander("데이터 미리보기"):
                st.write(sst_data)

        with tab2:
            st.subheader(f"{selected_date:%Y년 %m월 %d일} 해수면 온도 평년 편차 지도")
            if anomaly_data is not None:
                fig_anomaly = create_anomaly_map_figure(anomaly_data, selected_date)
                if fig_anomaly:
                    st.pyplot(fig_anomaly, clear_figure=True)
                with st.expander("편차 데이터 미리보기"):
                    st.write(anomaly_data)
            else:
                st.warning("평년 편차 데이터를 계산할 수 없습니다.")
    
    elif sst_data is not None:
        st.warning("선택하신 날짜에 해당하는 데이터가 없습니다. 다른 날짜를 선택해 주세요.")
    else:
        st.stop()