import numpy as np
import pandas as pd

'''
this code has been updated for the feature names that come from
flat TTrees.  for previous version, see feb2019 working directory.
features are:

run,event,rftime,
kin_chisq,kin_ndf,
kp_beta_time,kp_chisq_time,kp_ndf_time,kp_ndf_trk,kp_chisq_trk,kp_ndf_dedx,kp_chisq_dedx,kp_dedx_cdc,kp_dedx_fdc,kp_dedx_tof,kp_dedx_st,kp_ebcal,kp_eprebcal,kp_efcal,kp_bcal_delphi,kp_bcal_delz,kp_fcal_doca,
p_beta_time,p_chisq_time,p_ndf_time,p_ndf_trk,p_chisq_trk,p_ndf_dedx,p_chisq_dedx,p_dedx_cdc,p_dedx_fdc,p_dedx_tof,p_dedx_st,p_ebcal,p_eprebcal,p_efcal,p_bcal_delphi,p_bcal_delz,p_fcal_doca,
mum_beta_time,mum_chisq_time,mum_ndf_time,mum_ndf_trk,mum_chisq_trk,mum_ndf_dedx,mum_chisq_dedx,mum_dedx_cdc,mum_dedx_fdc,mum_dedx_tof,mum_dedx_st,mum_ebcal,mum_eprebcal,mum_efcal,mum_bcal_delphi,mum_bcal_delz,mum_fcal_doca,
beam_e,
kp_p4meas_px,kp_p4meas_py,kp_p4meas_pz,kp_p4meas_e,kp_p4kin_px,kp_p4kin_py,kp_p4kin_pz,kp_p4kin_e,
p_p4meas_px,p_p4meas_py,p_p4meas_pz,p_p4meas_e,p_p4kin_px,p_p4kin_py,p_p4kin_pz,p_p4kin_e,
mum_p4meas_px,mum_p4meas_py,mum_p4meas_pz,mum_p4meas_e,mum_p4kin_px,mum_p4kin_py,mum_p4kin_pz,mum_p4kin_e,
missneut_p4kin_px,missneut_p4kin_py,missneut_p4kin_pz,missneut_p4kin_e,
production_x4_x,production_x4_y,production_x4_z,production_x4_t,
lambda_x4kin_x,lambda_x4kin_y,lambda_x4kin_z,lambda_x4kin_t
'''

def sld_add_features(df):
    # calculate new columns
    print('calculating new features...\n')
    calc_drops = []

    df['kin_chisq_ndf'] = df['kin_chisq'] / df['kin_ndf']

    df['kp_chisq_ndf_trk'] = df['kp_chisq_trk'] / df['kp_ndf_trk']
    df['kp_chisq_ndf_time'] = df['kp_chisq_time'] / df['kp_ndf_time']
    df['kp_chisq_ndf_dedx'] = df['kp_chisq_dedx'] / df['kp_ndf_dedx']

    df['p_chisq_ndf_trk'] = df['p_chisq_trk'] / df['p_ndf_trk']
    df['p_chisq_ndf_time'] = df['p_chisq_time'] / df['p_ndf_time']
    df['p_chisq_ndf_dedx'] = df['p_chisq_dedx'] / df['p_ndf_dedx']

    df['mum_chisq_ndf_trk'] = df['mum_chisq_trk'] / df['mum_ndf_trk']
    df['mum_chisq_ndf_time'] = df['mum_chisq_time'] / df['mum_ndf_time']
    #df['mum_chisq_ndf_dedx'] = df['mum_chisq_dedx'] / df['mum_ndf_dedx']

    # cylindrical coords for vector quantites
    df['kp_p4kin_perp'] = np.sqrt(df['kp_p4kin_py']**2 + df['kp_p4kin_pz']**2)
    df['kp_p4kin_phi'] = np.arctan2(df['kp_p4kin_py'], df['kp_p4kin_px'])

    df['p_p4kin_perp'] = np.sqrt(df['p_p4kin_py']**2 + df['p_p4kin_pz']**2)
    df['p_p4kin_phi'] = np.arctan2(df['p_p4kin_py'], df['p_p4kin_px'])

    df['mum_p4kin_perp'] = np.sqrt(df['mum_p4kin_py']**2 + df['mum_p4kin_pz']**2)
    df['mum_p4kin_phi'] = np.arctan2(df['mum_p4kin_py'], df['mum_p4kin_px'])

    df['missneut_p4kin_perp'] = np.sqrt(df['missneut_p4kin_py']**2 + df['missneut_p4kin_pz']**2)
    df['missneut_p4kin_phi'] = np.arctan2(df['missneut_p4kin_py'], df['missneut_p4kin_px'])

    # missing mass off of K+
    df['mmoffk2'] = ((df['beam_e'] + 0.938272 - df['kp_p4kin_e'])**2
                     - (df['kp_p4kin_px']**2 + df['kp_p4kin_py']**2 + (df['beam_e'] - df['kp_p4kin_pz'])**2))

    df['magp_k'] = np.sqrt(df['kp_p4kin_px']**2 + df['kp_p4kin_py']**2 + df['kp_p4kin_pz']**2)
    df['magp_muon'] = np.sqrt(df['mum_p4kin_px']**2 + df['mum_p4kin_py']**2 + df['mum_p4kin_pz']**2)
    df['magp_proton'] = np.sqrt(df['p_p4kin_px']**2 + df['p_p4kin_py']**2 + df['p_p4kin_pz']**2)
    df['magp_miss'] = np.sqrt(df['missneut_p4kin_px']**2 + df['missneut_p4kin_py']**2 + df['missneut_p4kin_pz']**2)

    df['costheta_k'] = df['kp_p4kin_pz'] / df['magp_k']

    df['invm2_pmu'] = ((df['p_p4kin_e'] + df['mum_p4kin_e'])**2
                       - ((df['p_p4kin_px'] + df['mum_p4kin_px'])**2
                       + (df['p_p4kin_py'] + df['mum_p4kin_py'])**2
                       + (df['p_p4kin_pz'] + df['mum_p4kin_pz'])**2))
    df['invm2_munu'] = ((df['missneut_p4kin_e'] + df['mum_p4kin_e'])**2
                        - ((df['missneut_p4kin_px'] + df['mum_p4kin_px'])**2
                        + (df['missneut_p4kin_py'] + df['mum_p4kin_py'])**2
                        + (df['missneut_p4kin_pz'] + df['mum_p4kin_pz'])**2))

    df['lambda_px'] = -df['kp_p4kin_px']
    df['lambda_py'] = -df['kp_p4kin_py']
    df['lambda_pz'] = df['beam_e'] - df['kp_p4kin_pz']
    df['magp_lambda'] = np.sqrt(df['lambda_px']**2 + df['lambda_py']**2 + df['lambda_pz']**2)
    df['lambda_costheta'] = df['lambda_pz'] / df['magp_lambda']

    df['lambda_perp'] = np.sqrt(df['lambda_py']**2 + df['lambda_pz']**2)
    df['lambda_phi'] = np.arctan2(df['lambda_py'], df['lambda_px'])


    # df['magp_lambda_sum'] = np.sqrt((df['mum_p4kin_px'] + df['p_p4kin_px'] + df['miss_px'])**2 +
    #                                 (df['mum_p4kin_py'] + df['p_p4kin_py'] + df['miss_py'])**2 +
    #                                 (df['mum_p4kin_pz'] + df['p_p4kin_pz'] + df['miss_pz'])**2)

    # angle between muon/proton momentum and Lambda momentum
    df['muon_open_costheta'] = (df['mum_p4kin_px']*df['lambda_px'] +
                                df['mum_p4kin_py']*df['lambda_py'] +
                                df['mum_p4kin_pz']*df['lambda_pz']) / df['magp_muon'] / df['magp_lambda']

    df['proton_open_costheta'] = (df['p_p4kin_px']*df['lambda_px'] +
                                  df['p_p4kin_py']*df['lambda_py'] +
                                  df['p_p4kin_pz']*df['lambda_pz']) / df['magp_proton'] / df['magp_lambda']

    # mandelstam t
    df['mandel_t'] = ((df['beam_e'] - df['kp_p4kin_e'])**2
                      - (0 + 0 + (df['beam_e'] - df['kp_p4kin_pz'])**2))

    df['muon_pt_x'] = (df['mum_p4kin_px'] -
                       df['magp_muon']*df['muon_open_costheta']*df['lambda_px']/df['magp_lambda'])
    df['muon_pt_y'] = (df['mum_p4kin_py'] -
                       df['magp_muon']*df['muon_open_costheta']*df['lambda_py']/df['magp_lambda'])
    df['muon_pt_z'] = (df['mum_p4kin_pz'] -
                       df['magp_muon']*df['muon_open_costheta']*df['lambda_pz']/df['magp_lambda'])
    df['muon_pt'] = np.sqrt(df['muon_pt_x']**2 + df['muon_pt_y']**2 + df['muon_pt_z']**2)

    df['proton_pt_x'] = (df['p_p4kin_px'] -
                         df['magp_proton']*df['proton_open_costheta']*df['lambda_px']/df['magp_lambda'])
    df['proton_pt_y'] = (df['p_p4kin_py'] -
                         df['magp_proton']*df['proton_open_costheta']*df['lambda_py']/df['magp_lambda'])
    df['proton_pt_z'] = (df['p_p4kin_pz'] -
                         df['magp_proton']*df['proton_open_costheta']*df['lambda_pz']/df['magp_lambda'])
    df['proton_pt'] = np.sqrt(df['proton_pt_x']**2 + df['proton_pt_y']**2 + df['proton_pt_z']**2)

    # canter momentum imbalance
    df['canter_t'] = df['muon_pt'] - df['proton_pt']

    df['canter_costheta'] = (df['proton_pt_x']*df['muon_pt_x'] +
                             df['proton_pt_y']*df['muon_pt_y'] +
                             df['proton_pt_z']*df['muon_pt_z']) / df['proton_pt'] / df['muon_pt']

    #df['lambda_flight_len'] = np.sqrt((df['lambda_x4_x'] - df['production_x4_x'])**2 +
    #                                  (df['lambda_x4kin_y'] - df['production_x4_y'])**2 +
    #                                  (df['lambda_x4kin_z'] - df['production_x4_z'])**2)

    # lambda flight discrep
    df['lam_p3_unit_x'] = df['lambda_px'] / df['magp_lambda']
    df['lam_p3_unit_y'] = df['lambda_py'] / df['magp_lambda']
    df['lam_p3_unit_z'] = df['lambda_pz'] / df['magp_lambda']
    calc_drops.append('lam_p3_unit_x')
    calc_drops.append('lam_p3_unit_y')
    calc_drops.append('lam_p3_unit_z')

    df['v1v2_diff_x'] = (df['production_x4_x'] - df['lambda_x4kin_x'])
    df['v1v2_diff_y'] = (df['production_x4_y'] - df['lambda_x4kin_y'])
    df['v1v2_diff_z'] = (df['production_x4_z'] - df['lambda_x4kin_z'])
    df['lambda_flight_len'] = np.sqrt(df['v1v2_diff_x']**2 +
                                      df['v1v2_diff_y']**2 +
                                      df['v1v2_diff_z']**2)
    calc_drops.append('v1v2_diff_x')
    calc_drops.append('v1v2_diff_y')
    calc_drops.append('v1v2_diff_z')

    df['disc_dot'] = (df['v1v2_diff_x']*df['lam_p3_unit_x']
                      + df['v1v2_diff_y']*df['lam_p3_unit_y']
                      + df['v1v2_diff_z']*df['lam_p3_unit_z'])
    calc_drops.append('disc_dot')

    df['lam_flight_disc_x'] = df['v1v2_diff_x'] - df['disc_dot']*df['lam_p3_unit_x']
    df['lam_flight_disc_y'] = df['v1v2_diff_y'] - df['disc_dot']*df['lam_p3_unit_y']
    df['lam_flight_disc_z'] = df['v1v2_diff_z'] - df['disc_dot']*df['lam_p3_unit_z']
    calc_drops.append('lam_flight_disc_x')
    calc_drops.append('lam_flight_disc_y')
    calc_drops.append('lam_flight_disc_z')

    df['lam_flight_disc'] = np.sqrt(df['lam_flight_disc_x']**2
                                    + df['lam_flight_disc_y']**2
                                    + df['lam_flight_disc_z']**2)

    df['v2v1_costheta'] = -df['v1v2_diff_z'] / df['lambda_flight_len']
    df['v2v1_plam_ct_diff'] = (-df['v1v2_diff_z'] / df['lambda_flight_len'] -
                              df['lambda_costheta'])

    #df['log_kinfit_cl'] = np.log10(df['kinfit_cl'])

    df.drop(calc_drops, axis=1, inplace=True)

    return df
