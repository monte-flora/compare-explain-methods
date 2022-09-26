# Predictor columns. Order matters! Wouldn't touch this.
PREDICTOR_COLUMNS = ['dllwave_flux', 'dwpt2m', 'fric_vel', 'gflux', 'high_cloud',
            'lat_hf', 'low_cloud', 'mid_cloud', 'sat_irbt', 'sens_hf',
            'sfcT_hrs_ab_frez', 'sfcT_hrs_bl_frez', 'sfc_rough', 'sfc_temp',
            'swave_flux', 'temp2m', 'tmp2m_hrs_ab_frez', 'tmp2m_hrs_bl_frez',
            'tot_cloud', 'uplwav_flux', 'vbd_flux', 'vdd_flux', 'wind10m',
            'date_marker', 'urban','rural','d_ground','d_rad_d','d_rad_u',
            'hrrr_dT']

units = ['W m$^{-2}$', '$^{\circ}$C', 'm s$^{-1}$', 'W m$^{-2}$', '%', 'W m$^{-2}$', '%', '%', 
         '$^{\circ}$C', 'W m$^{-2}$', 'hrs', 'hrs', 'unitless','$^{\circ}$C', 'W m$^{-2}$', '$^{\circ}$C', 
         'hrs', 'hrs', '%', 'W m$^{-2}$', 'W m$^{-2}$', 'W m$^{-2}$', 'm s$^{-1}$', 'days', 'unitless', 
         'unitless', 'W m$^{-2}$', 'W m$^{-2}$', 'W m$^{-2}$', '$^{\circ}$C']

# The target variable
TARGET_COLUMN = 'cat_rt'

# Dictionary that maps predictor columns to pretty names
FIGURE_MAPPINGS = {
    'dllwave_flux': '$\lambda_{\downarrow}$',
    'dwpt2m': '$T_{d}$',
    'fric_vel': '$V_{fric}$',
    'gflux': 'G',
    'high_cloud': '$C_{high}$',
    'lat_hf':'$L_{hf}$',
    'low_cloud': '$C_{low}$',
    'mid_cloud': '$C_{mid}$',
    'sat_irbt':'$T_{irbt}$',
    'sens_hf': '$S_{hf}$',
    'sfcT_hrs_ab_frez': 'Hours T$_{sfc}$ > 0$^{\circ}$C',
    'sfcT_hrs_bl_frez': 'Hours T$_{sfc}$ $\leq$ 0$^{\circ}$C',
    'sfc_rough': '$S_{R}$',
    'sfc_temp': '$T_{sfc}$',
    'swave_flux': 'S',
    'temp2m': '$T_{2m}$',
    'tmp2m_hrs_ab_frez':'Hours T$_{2m}$ > 0$^{\circ}$C',
    'tmp2m_hrs_bl_frez':'Hours T$_{2m}$ $\leq$ 0$^{\circ}$C',
    'tot_cloud': '$C_{total}$',
    'uplwav_flux':r'$\lambda_{\uparrow}$',
    'vbd_flux': '$V_{bd}$',
    'vdd_flux': '$V_{dd}$',
    'wind10m': '$U_{10m}$',
    'date_marker': 'Date Marker',
    'urban': '$L_{urban}$',
    'rural': '$L_{rural}$',
    'd_ground': '$G_{diff}$',
    'd_rad_d': '$dR_{down}$',
    'd_rad_u': '$dR_{up}$',
    'hrrr_dT': '$dT_{hrrr}$'
}

COLOR_DICT = { 'dllwave_flux':'xkcd:light light green',
              'dwpt2m': 'xkcd:powder blue',
              'gflux':'xkcd:light light green',
              'high_cloud':'xkcd:light periwinkle',
              'lat_hf':'xkcd:light light green',
              'low_cloud':'xkcd:light periwinkle',
              'mid_cloud':'xkcd:light periwinkle',
              'sat_irbt':'xkcd:light periwinkle',
              'sens_hf':'xkcd:light light green',
            'sfcT_hrs_ab_frez':'xkcd:powder blue',
            'sfcT_hrs_bl_frez':'xkcd:powder blue',
            'sfc_rough':'xkcd:orangish',
            'sfc_temp':'xkcd:powder blue',
            'swave_flux':'xkcd:light light green',
            'temp2m':'xkcd:powder blue',
            'tmp2m_hrs_ab_frez':'xkcd:powder blue',
            'tmp2m_hrs_bl_frez':'xkcd:powder blue',
            'tot_cloud':'xkcd:light periwinkle',
            'uplwav_flux':'xkcd:light light green',
            'vbd_flux':'xkcd:light light green',
            'vdd_flux':'xkcd:light light green',
            'wind10m':'xkcd:orangish',
            'date_marker':'xkcd:orangish',
            'urban':'xkcd:orangish',
            'rural':'xkcd:orangish',
            'd_ground':'xkcd:light light green',
            'd_rad_d':'xkcd:light light green',
            'd_rad_u':'xkcd:light light green',
            'hrrr_dT':'xkcd:powder blue', 
             'fric_vel' : 'xkcd:orangish'}

UNITS = {c : u for c,u in zip(PREDICTOR_COLUMNS , units)}


PIMP_MAPPINGS = {
    'dllwave_flux': 'Downward longwave Rad. flux',
    'dwpt2m': 'Dewpoint Temperature',
    'fric_vel': '$V_{fric}$',
    'gflux': 'G',
    'high_cloud': '$C_{high}$',
    'lat_hf':'Latent Heat Flux',
    'low_cloud': 'Low cloud cover percentage',
    'mid_cloud': '$C_{mid}$',
    'sat_irbt':'Simulated Brightness Temp.',
    'sens_hf': '$S_{hf}$',
    'sfcT_hrs_ab_frez': 'Hours $T_{sfc}$ $> $0$\circ$C',
    'sfcT_hrs_bl_frez': 'Hours $T_{sfc}$ $<= $0$\circ$C',
    'sfc_rough': '$S_{R}$',
    'sfc_temp': 'Surface Temperature',
    'swave_flux': 'S',
    'temp2m': '$T_{2m}$',
    'tmp2m_hrs_ab_frez':'Hours $T_{2m}$ $> $0$\circ$C',
    'tmp2m_hrs_bl_frez':'Hours $T_{2m}$ $<= $0$\circ$C',
    'tot_cloud': '$C_{total}$',
    'uplwav_flux':r'$\lambda_{\uparrow}$',
    'vbd_flux': '$V_{bd}$',
    'vdd_flux': '$V_{dd}$',
    'wind10m': '$U_{10m}$',
    'date_marker': 'Date Marker',
    'urban': '$L_{urban}$',
    'rural': '$L_{rural}$',
    'd_ground': '$G_{diff}$',
    'd_rad_d': '$dR_{down}$',
    'd_rad_u': '$dR_{up}$',
    'hrrr_dT': '$dT_{hrrr}$'
}


