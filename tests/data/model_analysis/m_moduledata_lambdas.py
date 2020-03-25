from numbers import Real
from random import random
from delphi.translators.for2py.strings import *
import numpy as np
import delphi.translators.for2py.math_ext as math

def mini_ModuleDefs__get_control__assign__control_arg__0(control: controltype):
    return control

def mini_ModuleDefs__put_control__assign__save_data__control___1(control_arg: controltype, save_data: transfertype):
    save_data.control = control_arg
    return save_data.control

def mini_ModuleDefs__get_iswitch__assign__iswitch_arg__0(iswitch: switchtype):
    return iswitch

def mini_ModuleDefs__put_iswitch__assign__save_data__iswitch___1(iswitch_arg: switchtype, save_data: transfertype):
    save_data.iswitch = iswitch_arg
    return save_data.iswitch

def mini_ModuleDefs__get_output__assign__output_arg__0(output: outputtype):
    return output

def mini_ModuleDefs__put_output__assign__save_data__output___1(output_arg: outputtype, save_data: transfertype):
    save_data.output = output_arg
    return save_data.output

def mini_ModuleDefs__get_soilprop__assign__soil_arg__0(soilprop: soiltype):
    return soilprop

def mini_ModuleDefs__put_soilprop__assign__save_data__soilprop___1(soil_arg: soiltype, save_data: transfertype):
    save_data.soilprop = soil_arg
    return save_data.soilprop

def mini_ModuleDefs__get_real__assign__value__0():
    return 0.0

def mini_ModuleDefs__get_real__assign__err__0():
    return False

def mini_ModuleDefs__get_real__condition__IF_0__0(modulename):
    return (modulename == "spam")

def mini_ModuleDefs__get_real__assign__spam_tmp___1(spam: spamtype):
    return spam

def mini_ModuleDefs__get_real__condition__IF_None__1(varname):
    return (varname == "agefac")

def mini_ModuleDefs__get_real__assign__value__1(agefac: Real):
    return agefac

def mini_ModuleDefs__get_real__decision__value__2(value_0: Real, value_1: Real, IF_None_1: int):
    return np.where(IF_None_1, value_1, value_0)

def mini_ModuleDefs__get_real__decision__agefac__0(agefac_0: Real, agefac_1: Real, IF_None_1: int):
    return np.where(IF_None_1, agefac_1, agefac_0)

def mini_ModuleDefs__get_real__condition__IF_None__2(varname):
    return (varname == "pg")

def mini_ModuleDefs__get_real__assign__value__3(pg: Real):
    return pg

def mini_ModuleDefs__get_real__decision__value__4(value_0: Real, value_1: Real, IF_None_2: int):
    return np.where(IF_None_2, value_1, value_0)

def mini_ModuleDefs__get_real__decision__pg__0(pg_0: int, pg_1: int, IF_None_2: int):
    return np.where(IF_None_2, pg_1, pg_0)

def mini_ModuleDefs__get_real__condition__IF_None__3(varname):
    return (varname == "cef")

def mini_ModuleDefs__get_real__assign__value__5(cef: Real):
    return cef

def mini_ModuleDefs__get_real__decision__value__6(value_0: Real, value_1: Real, IF_None_3: int):
    return np.where(IF_None_3, value_1, value_0)

def mini_ModuleDefs__get_real__decision__cef__0(cef_0: Real, cef_1: Real, IF_None_3: int):
    return np.where(IF_None_3, cef_1, cef_0)

def mini_ModuleDefs__get_real__condition__IF_None__4(varname):
    return (varname == "cem")

def mini_ModuleDefs__get_real__assign__value__7(cem: Real):
    return cem

def mini_ModuleDefs__get_real__decision__value__8(value_0: Real, value_1: Real, IF_None_4: int):
    return np.where(IF_None_4, value_1, value_0)

def mini_ModuleDefs__get_real__decision__cem__0(cem_0: Real, cem_1: Real, IF_None_4: int):
    return np.where(IF_None_4, cem_1, cem_0)

def mini_ModuleDefs__get_real__condition__IF_None__5(varname):
    return (varname == "ceo")

def mini_ModuleDefs__get_real__assign__value__9(ceo: Real):
    return ceo

def mini_ModuleDefs__get_real__decision__value__10(value_0: Real, value_1: Real, IF_None_5: int):
    return np.where(IF_None_5, value_1, value_0)

def mini_ModuleDefs__get_real__decision__ceo__0(ceo_0: Real, ceo_1: Real, IF_None_5: int):
    return np.where(IF_None_5, ceo_1, ceo_0)

def mini_ModuleDefs__get_real__condition__IF_None__6(varname):
    return (varname == "cep")

def mini_ModuleDefs__get_real__assign__value__11(cep: Real):
    return cep

def mini_ModuleDefs__get_real__decision__value__12(value_0: Real, value_1: Real, IF_None_6: int):
    return np.where(IF_None_6, value_1, value_0)

def mini_ModuleDefs__get_real__decision__cep__0(cep_0: int, cep_1: int, IF_None_6: int):
    return np.where(IF_None_6, cep_1, cep_0)

def mini_ModuleDefs__get_real__condition__IF_None__7(varname):
    return (varname == "ces")

def mini_ModuleDefs__get_real__assign__value__13(ces: Real):
    return ces

def mini_ModuleDefs__get_real__decision__value__14(value_0: Real, value_1: Real, IF_None_7: int):
    return np.where(IF_None_7, value_1, value_0)

def mini_ModuleDefs__get_real__decision__ces__0(ces_0: Real, ces_1: Real, IF_None_7: int):
    return np.where(IF_None_7, ces_1, ces_0)

def mini_ModuleDefs__get_real__condition__IF_None__8(varname):
    return (varname == "cet")

def mini_ModuleDefs__get_real__assign__value__15(cet: Real):
    return cet

def mini_ModuleDefs__get_real__decision__value__16(value_0: Real, value_1: Real, IF_None_8: int):
    return np.where(IF_None_8, value_1, value_0)

def mini_ModuleDefs__get_real__decision__cet__0(cet_0: Real, cet_1: Real, IF_None_8: int):
    return np.where(IF_None_8, cet_1, cet_0)

def mini_ModuleDefs__get_real__condition__IF_None__9(varname):
    return (varname == "ef")

def mini_ModuleDefs__get_real__assign__value__17(ef: Real):
    return ef

def mini_ModuleDefs__get_real__decision__value__18(value_0: Real, value_1: Real, IF_None_9: int):
    return np.where(IF_None_9, value_1, value_0)

def mini_ModuleDefs__get_real__decision__ef__0(ef_0: Real, ef_1: Real, IF_None_9: int):
    return np.where(IF_None_9, ef_1, ef_0)

def mini_ModuleDefs__get_real__condition__IF_None__10(varname):
    return (varname == "em")

def mini_ModuleDefs__get_real__assign__value__19(em: Real):
    return em

def mini_ModuleDefs__get_real__decision__value__20(value_0: Real, value_1: Real, IF_None_10: int):
    return np.where(IF_None_10, value_1, value_0)

def mini_ModuleDefs__get_real__decision__em__0(em_0: Real, em_1: Real, IF_None_10: int):
    return np.where(IF_None_10, em_1, em_0)

def mini_ModuleDefs__get_real__condition__IF_None__11(varname):
    return (varname == "eo")

def mini_ModuleDefs__get_real__assign__value__21(eo: Real):
    return eo

def mini_ModuleDefs__get_real__decision__value__22(value_0: Real, value_1: Real, IF_None_11: int):
    return np.where(IF_None_11, value_1, value_0)

def mini_ModuleDefs__get_real__decision__eo__0(eo_0: Real, eo_1: Real, IF_None_11: int):
    return np.where(IF_None_11, eo_1, eo_0)

def mini_ModuleDefs__get_real__condition__IF_None__12(varname):
    return (varname == "ep")

def mini_ModuleDefs__get_real__assign__value__23(ep: Real):
    return ep

def mini_ModuleDefs__get_real__decision__value__24(value_0: Real, value_1: Real, IF_None_12: int):
    return np.where(IF_None_12, value_1, value_0)

def mini_ModuleDefs__get_real__decision__ep__0(ep_0: int, ep_1: int, IF_None_12: int):
    return np.where(IF_None_12, ep_1, ep_0)

def mini_ModuleDefs__get_real__condition__IF_None__13(varname):
    return (varname == "es")

def mini_ModuleDefs__get_real__assign__value__25(es: Real):
    return es

def mini_ModuleDefs__get_real__decision__value__26(value_0: Real, value_1: Real, IF_None_13: int):
    return np.where(IF_None_13, value_1, value_0)

def mini_ModuleDefs__get_real__decision__es__0(es_0: Real, es_1: Real, IF_None_13: int):
    return np.where(IF_None_13, es_1, es_0)

def mini_ModuleDefs__get_real__condition__IF_None__14(varname):
    return (varname == "et")

def mini_ModuleDefs__get_real__assign__value__27(et: Real):
    return et

def mini_ModuleDefs__get_real__decision__value__28(value_0: Real, value_1: Real, IF_None_14: int):
    return np.where(IF_None_14, value_1, value_0)

def mini_ModuleDefs__get_real__decision__et__0(et_0: Real, et_1: Real, IF_None_14: int):
    return np.where(IF_None_14, et_1, et_0)

def mini_ModuleDefs__get_real__condition__IF_None__15(varname):
    return (varname == "eop")

def mini_ModuleDefs__get_real__assign__value__29(eop: Real):
    return eop

def mini_ModuleDefs__get_real__decision__value__30(value_0: Real, value_1: Real, IF_None_15: int):
    return np.where(IF_None_15, value_1, value_0)

def mini_ModuleDefs__get_real__decision__eop__0(eop_0: int, eop_1: int, IF_None_15: int):
    return np.where(IF_None_15, eop_1, eop_0)

def mini_ModuleDefs__get_real__condition__IF_None__16(varname):
    return (varname == "evap")

def mini_ModuleDefs__get_real__assign__value__31(evap: Real):
    return evap

def mini_ModuleDefs__get_real__decision__value__32(value_0: Real, value_1: Real, IF_None_16: int):
    return np.where(IF_None_16, value_1, value_0)

def mini_ModuleDefs__get_real__decision__evap__0(evap_0: int, evap_1: int, IF_None_16: int):
    return np.where(IF_None_16, evap_1, evap_0)

def mini_ModuleDefs__get_real__condition__IF_None__17(varname):
    return (varname == "refet")

def mini_ModuleDefs__get_real__assign__value__33(refet: Real):
    return refet

def mini_ModuleDefs__get_real__decision__value__34(value_0: Real, value_1: Real, IF_None_17: int):
    return np.where(IF_None_17, value_1, value_0)

def mini_ModuleDefs__get_real__decision__refet__0(refet_0: Real, refet_1: Real, IF_None_17: int):
    return np.where(IF_None_17, refet_1, refet_0)

def mini_ModuleDefs__get_real__condition__IF_None__18(varname):
    return (varname == "skc")

def mini_ModuleDefs__get_real__assign__value__35(skc: Real):
    return skc

def mini_ModuleDefs__get_real__decision__value__36(value_0: Real, value_1: Real, IF_None_18: int):
    return np.where(IF_None_18, value_1, value_0)

def mini_ModuleDefs__get_real__decision__skc__0(skc_0: Real, skc_1: Real, IF_None_18: int):
    return np.where(IF_None_18, skc_1, skc_0)

def mini_ModuleDefs__get_real__condition__IF_None__19(varname):
    return (varname == "kcbmin")

def mini_ModuleDefs__get_real__assign__value__37(kcbmin: Real):
    return kcbmin

def mini_ModuleDefs__get_real__decision__value__38(value_0: Real, value_1: Real, IF_None_19: int):
    return np.where(IF_None_19, value_1, value_0)

def mini_ModuleDefs__get_real__decision__kcbmin__0(kcbmin_0: int, kcbmin_1: int, IF_None_19: int):
    return np.where(IF_None_19, kcbmin_1, kcbmin_0)

def mini_ModuleDefs__get_real__condition__IF_None__20(varname):
    return (varname == "kcbmax")

def mini_ModuleDefs__get_real__assign__value__39(kcbmax: Real):
    return kcbmax

def mini_ModuleDefs__get_real__decision__value__40(value_0: Real, value_1: Real, IF_None_20: int):
    return np.where(IF_None_20, value_1, value_0)

def mini_ModuleDefs__get_real__decision__kcbmax__0(kcbmax_0: Real, kcbmax_1: Real, IF_None_20: int):
    return np.where(IF_None_20, kcbmax_1, kcbmax_0)

def mini_ModuleDefs__get_real__condition__IF_None__21(varname):
    return (varname == "kcb")

def mini_ModuleDefs__get_real__assign__value__41(kcb: Real):
    return kcb

def mini_ModuleDefs__get_real__decision__value__42(value_0: Real, value_1: Real, IF_None_21: int):
    return np.where(IF_None_21, value_1, value_0)

def mini_ModuleDefs__get_real__decision__kcb__0(kcb_0: Real, kcb_1: Real, IF_None_21: int):
    return np.where(IF_None_21, kcb_1, kcb_0)

def mini_ModuleDefs__get_real__condition__IF_None__22(varname):
    return (varname == "ke")

def mini_ModuleDefs__get_real__assign__value__43(ke: Real):
    return ke

def mini_ModuleDefs__get_real__decision__value__44(value_0: Real, value_1: Real, IF_None_22: int):
    return np.where(IF_None_22, value_1, value_0)

def mini_ModuleDefs__get_real__decision__ke__0(ke_0: Real, ke_1: Real, IF_None_22: int):
    return np.where(IF_None_22, ke_1, ke_0)

def mini_ModuleDefs__get_real__condition__IF_None__23(varname):
    return (varname == "kc")

def mini_ModuleDefs__get_real__assign__value__45(kc: Real):
    return kc

def mini_ModuleDefs__get_real__assign__err__1():
    return True

def mini_ModuleDefs__get_real__decision__value__46(value_0: Real, value_1: Real, IF_None_23: int):
    return np.where(IF_None_23, value_1, value_0)

def mini_ModuleDefs__get_real__decision__err__2(err_0: bool, err_1: bool, IF_None_23: int):
    return np.where(IF_None_23, err_1, err_0)

def mini_ModuleDefs__get_real__decision__kc__0(kc_0: Real, kc_1: Real, IF_None_23: int):
    return np.where(IF_None_23, kc_1, kc_0)

def mini_ModuleDefs__get_real__decision__value__47(value_0: Real, value_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, value_1, value_0)

def mini_ModuleDefs__get_real__decision__cet__1(cet_0: Real, cet_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, cet_1, cet_0)

def mini_ModuleDefs__get_real__decision__eop__1(eop_0: int, eop_1: int, IF_0_0: bool):
    return np.where(IF_0_0, eop_1, eop_0)

def mini_ModuleDefs__get_real__decision__agefac__1(agefac_0: Real, agefac_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, agefac_1, agefac_0)

def mini_ModuleDefs__get_real__decision__kcbmin__1(kcbmin_0: int, kcbmin_1: int, IF_0_0: bool):
    return np.where(IF_0_0, kcbmin_1, kcbmin_0)

def mini_ModuleDefs__get_real__decision__ceo__1(ceo_0: Real, ceo_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, ceo_1, ceo_0)

def mini_ModuleDefs__get_real__decision__err__3(err_0: bool, err_1: bool, IF_0_0: bool):
    return np.where(IF_0_0, err_1, err_0)

def mini_ModuleDefs__get_real__decision__ke__1(ke_0: Real, ke_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, ke_1, ke_0)

def mini_ModuleDefs__get_real__decision__ep__1(ep_0: int, ep_1: int, IF_0_0: bool):
    return np.where(IF_0_0, ep_1, ep_0)

def mini_ModuleDefs__get_real__decision__kcbmax__1(kcbmax_0: Real, kcbmax_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, kcbmax_1, kcbmax_0)

def mini_ModuleDefs__get_real__decision__cep__1(cep_0: int, cep_1: int, IF_0_0: bool):
    return np.where(IF_0_0, cep_1, cep_0)

def mini_ModuleDefs__get_real__decision__spam__0(spam_0: int, spam_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_1, spam_0)

def mini_ModuleDefs__get_real__decision__pg__1(pg_0: int, pg_1: int, IF_0_0: bool):
    return np.where(IF_0_0, pg_1, pg_0)

def mini_ModuleDefs__get_real__decision__ef__1(ef_0: Real, ef_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, ef_1, ef_0)

def mini_ModuleDefs__get_real__decision__cem__1(cem_0: Real, cem_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, cem_1, cem_0)

def mini_ModuleDefs__get_real__decision__cef__1(cef_0: Real, cef_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, cef_1, cef_0)

def mini_ModuleDefs__get_real__decision__em__1(em_0: Real, em_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, em_1, em_0)

def mini_ModuleDefs__get_real__decision__eo__1(eo_0: Real, eo_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, eo_1, eo_0)

def mini_ModuleDefs__get_real__decision__evap__1(evap_0: int, evap_1: int, IF_0_0: bool):
    return np.where(IF_0_0, evap_1, evap_0)

def mini_ModuleDefs__get_real__decision__es__1(es_0: Real, es_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, es_1, es_0)

def mini_ModuleDefs__get_real__decision__refet__1(refet_0: Real, refet_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, refet_1, refet_0)

def mini_ModuleDefs__get_real__decision__IF_None__24(IF_None_0: int, IF_None_1: int, IF_0_0: bool):
    return np.where(IF_0_0, IF_None_1, IF_None_0)

def mini_ModuleDefs__get_real__decision__kcb__1(kcb_0: Real, kcb_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, kcb_1, kcb_0)

def mini_ModuleDefs__get_real__decision__et__1(et_0: Real, et_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, et_1, et_0)

def mini_ModuleDefs__get_real__decision__skc__1(skc_0: Real, skc_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, skc_1, skc_0)

def mini_ModuleDefs__get_real__decision__ces__1(ces_0: Real, ces_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, ces_1, ces_0)

def mini_ModuleDefs__get_real__decision__kc__1(kc_0: Real, kc_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, kc_1, kc_0)

def mini_ModuleDefs__get_real__condition__IF_0__1(varname):
    return (varname == "plant")

def mini_ModuleDefs__get_real__assign__plant_tmp___1(plant: planttype):
    return plant

def mini_ModuleDefs__get_real__condition__IF_0__2(varname):
    return (varname == "biomas")

def mini_ModuleDefs__get_real__assign__value__48(biomas: Real):
    return biomas

def mini_ModuleDefs__get_real__decision__value__49(value_0: Real, value_1: Real, IF_0_2: bool):
    return np.where(IF_0_2, value_1, value_0)

def mini_ModuleDefs__get_real__decision__biomas__0(biomas_0: Real, biomas_1: Real, IF_0_2: bool):
    return np.where(IF_0_2, biomas_1, biomas_0)

def mini_ModuleDefs__get_real__condition__IF_0__3(varname):
    return (varname == "canht")

def mini_ModuleDefs__get_real__assign__value__50(canht: Real):
    return canht

def mini_ModuleDefs__get_real__decision__value__51(value_0: Real, value_1: Real, IF_0_3: bool):
    return np.where(IF_0_3, value_1, value_0)

def mini_ModuleDefs__get_real__decision__canht__0(canht_0: int, canht_1: int, IF_0_3: bool):
    return np.where(IF_0_3, canht_1, canht_0)

def mini_ModuleDefs__get_real__condition__IF_0__4(varname):
    return (varname == "canwh")

def mini_ModuleDefs__get_real__assign__value__52(canwh: Real):
    return canwh

def mini_ModuleDefs__get_real__decision__value__53(value_0: Real, value_1: Real, IF_0_4: bool):
    return np.where(IF_0_4, value_1, value_0)

def mini_ModuleDefs__get_real__decision__canwh__0(canwh_0: int, canwh_1: int, IF_0_4: bool):
    return np.where(IF_0_4, canwh_1, canwh_0)

def mini_ModuleDefs__get_real__condition__IF_0__5(varname):
    return (varname == "dxr57")

def mini_ModuleDefs__get_real__assign__value__54(dxr57: Real):
    return dxr57

def mini_ModuleDefs__get_real__decision__value__55(value_0: Real, value_1: Real, IF_0_5: bool):
    return np.where(IF_0_5, value_1, value_0)

def mini_ModuleDefs__get_real__decision__dxr57__0(dxr57_0: Real, dxr57_1: Real, IF_0_5: bool):
    return np.where(IF_0_5, dxr57_1, dxr57_0)

def mini_ModuleDefs__get_real__condition__IF_0__6(varname):
    return (varname == "excess")

def mini_ModuleDefs__get_real__assign__value__56(excess: Real):
    return excess

def mini_ModuleDefs__get_real__decision__value__57(value_0: Real, value_1: Real, IF_0_6: bool):
    return np.where(IF_0_6, value_1, value_0)

def mini_ModuleDefs__get_real__decision__excess__0(excess_0: Real, excess_1: Real, IF_0_6: bool):
    return np.where(IF_0_6, excess_1, excess_0)

def mini_ModuleDefs__get_real__condition__IF_0__7(varname):
    return (varname == "pltpop")

def mini_ModuleDefs__get_real__assign__value__58(pltpop: Real):
    return pltpop

def mini_ModuleDefs__get_real__decision__value__59(value_0: Real, value_1: Real, IF_0_7: bool):
    return np.where(IF_0_7, value_1, value_0)

def mini_ModuleDefs__get_real__decision__pltpop__0(pltpop_0: int, pltpop_1: int, IF_0_7: bool):
    return np.where(IF_0_7, pltpop_1, pltpop_0)

def mini_ModuleDefs__get_real__condition__IF_0__8(varname):
    return (varname == "rnitp")

def mini_ModuleDefs__get_real__assign__value__60(rnitp: Real):
    return rnitp

def mini_ModuleDefs__get_real__decision__value__61(value_0: Real, value_1: Real, IF_0_8: bool):
    return np.where(IF_0_8, value_1, value_0)

def mini_ModuleDefs__get_real__decision__rnitp__0(rnitp_0: int, rnitp_1: int, IF_0_8: bool):
    return np.where(IF_0_8, rnitp_1, rnitp_0)

def mini_ModuleDefs__get_real__condition__IF_0__9(varname):
    return (varname == "slaad")

def mini_ModuleDefs__get_real__assign__value__62(slaad: Real):
    return slaad

def mini_ModuleDefs__get_real__decision__value__63(value_0: Real, value_1: Real, IF_0_9: bool):
    return np.where(IF_0_9, value_1, value_0)

def mini_ModuleDefs__get_real__decision__slaad__0(slaad_0: Real, slaad_1: Real, IF_0_9: bool):
    return np.where(IF_0_9, slaad_1, slaad_0)

def mini_ModuleDefs__get_real__condition__IF_0__10(varname):
    return (varname == "xpod")

def mini_ModuleDefs__get_real__assign__value__64(xpod: Real):
    return xpod

def mini_ModuleDefs__get_real__assign__err__4():
    return True

def mini_ModuleDefs__get_real__decision__value__65(value_0: Real, value_1: Real, IF_0_10: bool):
    return np.where(IF_0_10, value_1, value_0)

def mini_ModuleDefs__get_real__decision__xpod__0(xpod_0: int, xpod_1: int, IF_0_10: bool):
    return np.where(IF_0_10, xpod_1, xpod_0)

def mini_ModuleDefs__get_real__decision__err__5(err_0: bool, err_1: bool, IF_0_10: bool):
    return np.where(IF_0_10, err_1, err_0)

def mini_ModuleDefs__get_real__decision__value__66(value_0: Real, value_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, value_1, value_0)

def mini_ModuleDefs__get_real__decision__xpod__1(xpod_0: int, xpod_1: int, IF_0_1: bool):
    return np.where(IF_0_1, xpod_1, xpod_0)

def mini_ModuleDefs__get_real__decision__err__6(err_0: bool, err_1: bool, IF_0_1: bool):
    return np.where(IF_0_1, err_1, err_0)

def mini_ModuleDefs__get_real__decision__biomas__1(biomas_0: Real, biomas_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, biomas_1, biomas_0)

def mini_ModuleDefs__get_real__decision__excess__1(excess_0: Real, excess_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, excess_1, excess_0)

def mini_ModuleDefs__get_real__decision__canwh__1(canwh_0: int, canwh_1: int, IF_0_1: bool):
    return np.where(IF_0_1, canwh_1, canwh_0)

def mini_ModuleDefs__get_real__decision__plant__0(plant_0: int, plant_1: int, IF_0_1: bool):
    return np.where(IF_0_1, plant_1, plant_0)

def mini_ModuleDefs__get_real__decision__pltpop__1(pltpop_0: int, pltpop_1: int, IF_0_1: bool):
    return np.where(IF_0_1, pltpop_1, pltpop_0)

def mini_ModuleDefs__get_real__decision__canht__1(canht_0: int, canht_1: int, IF_0_1: bool):
    return np.where(IF_0_1, canht_1, canht_0)

def mini_ModuleDefs__get_real__decision__dxr57__1(dxr57_0: Real, dxr57_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, dxr57_1, dxr57_0)

def mini_ModuleDefs__get_real__decision__IF_0__11(IF_0_0: bool, IF_0_1: bool):
    return np.where(IF_0_1, IF_0_1, IF_0_0)

def mini_ModuleDefs__get_real__decision__rnitp__1(rnitp_0: int, rnitp_1: int, IF_0_1: bool):
    return np.where(IF_0_1, rnitp_1, rnitp_0)

def mini_ModuleDefs__get_real__decision__slaad__1(slaad_0: Real, slaad_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, slaad_1, slaad_0)

def mini_ModuleDefs__get_real__condition__IF_0__12(varname):
    return (varname == "mgmt")

def mini_ModuleDefs__get_real__assign__mgmt_tmp___1(mgmt: mgmttype):
    return mgmt

def mini_ModuleDefs__get_real__condition__IF_0__13(varname):
    return (varname == "effirr")

def mini_ModuleDefs__get_real__assign__value__67(effirr: Real):
    return effirr

def mini_ModuleDefs__get_real__decision__value__68(value_0: Real, value_1: Real, IF_0_13: bool):
    return np.where(IF_0_13, value_1, value_0)

def mini_ModuleDefs__get_real__decision__effirr__0(effirr_0: Real, effirr_1: Real, IF_0_13: bool):
    return np.where(IF_0_13, effirr_1, effirr_0)

def mini_ModuleDefs__get_real__condition__IF_0__14(varname):
    return (varname == "totir")

def mini_ModuleDefs__get_real__assign__value__69(totir: Real):
    return totir

def mini_ModuleDefs__get_real__decision__value__70(value_0: Real, value_1: Real, IF_0_14: bool):
    return np.where(IF_0_14, value_1, value_0)

def mini_ModuleDefs__get_real__decision__totir__0(totir_0: Real, totir_1: Real, IF_0_14: bool):
    return np.where(IF_0_14, totir_1, totir_0)

def mini_ModuleDefs__get_real__condition__IF_0__15(varname):
    return (varname == "toteffirr")

def mini_ModuleDefs__get_real__assign__value__71(toteffirr: Real):
    return toteffirr

def mini_ModuleDefs__get_real__decision__value__72(value_0: Real, value_1: Real, IF_0_15: bool):
    return np.where(IF_0_15, value_1, value_0)

def mini_ModuleDefs__get_real__decision__toteffirr__0(toteffirr_0: Real, toteffirr_1: Real, IF_0_15: bool):
    return np.where(IF_0_15, toteffirr_1, toteffirr_0)

def mini_ModuleDefs__get_real__condition__IF_0__16(varname):
    return (varname == "depir")

def mini_ModuleDefs__get_real__assign__value__73(depir: Real):
    return depir

def mini_ModuleDefs__get_real__decision__value__74(value_0: Real, value_1: Real, IF_0_16: bool):
    return np.where(IF_0_16, value_1, value_0)

def mini_ModuleDefs__get_real__decision__depir__0(depir_0: Array, depir_1: Array, IF_0_16: bool):
    return np.where(IF_0_16, depir_1, depir_0)

def mini_ModuleDefs__get_real__condition__IF_0__17(varname):
    return (varname == "irramt")

def mini_ModuleDefs__get_real__assign__value__75(irramt: Real):
    return irramt

def mini_ModuleDefs__get_real__decision__value__76(value_0: Real, value_1: Real, IF_0_17: bool):
    return np.where(IF_0_17, value_1, value_0)

def mini_ModuleDefs__get_real__decision__irramt__0(irramt_0: Real, irramt_1: Real, IF_0_17: bool):
    return np.where(IF_0_17, irramt_1, irramt_0)

def mini_ModuleDefs__get_real__condition__IF_0__18(varname):
    return (varname == "fernit")

def mini_ModuleDefs__get_real__assign__value__77(fernit: Real):
    return fernit

def mini_ModuleDefs__get_real__assign__err__7():
    return True

def mini_ModuleDefs__get_real__decision__value__78(value_0: Real, value_1: Real, IF_0_18: bool):
    return np.where(IF_0_18, value_1, value_0)

def mini_ModuleDefs__get_real__decision__fernit__0(fernit_0: int, fernit_1: int, IF_0_18: bool):
    return np.where(IF_0_18, fernit_1, fernit_0)

def mini_ModuleDefs__get_real__decision__err__8(err_0: bool, err_1: bool, IF_0_18: bool):
    return np.where(IF_0_18, err_1, err_0)

def mini_ModuleDefs__get_real__decision__value__79(value_0: Real, value_1: Real, IF_0_12: bool):
    return np.where(IF_0_12, value_1, value_0)

def mini_ModuleDefs__get_real__decision__fernit__1(fernit_0: int, fernit_1: int, IF_0_12: bool):
    return np.where(IF_0_12, fernit_1, fernit_0)

def mini_ModuleDefs__get_real__decision__mgmt__0(mgmt_0: mgmttype, mgmt_1: mgmttype, IF_0_12: bool):
    return np.where(IF_0_12, mgmt_1, mgmt_0)

def mini_ModuleDefs__get_real__decision__err__9(err_0: bool, err_1: bool, IF_0_12: bool):
    return np.where(IF_0_12, err_1, err_0)

def mini_ModuleDefs__get_real__decision__effirr__1(effirr_0: Real, effirr_1: Real, IF_0_12: bool):
    return np.where(IF_0_12, effirr_1, effirr_0)

def mini_ModuleDefs__get_real__decision__totir__1(totir_0: Real, totir_1: Real, IF_0_12: bool):
    return np.where(IF_0_12, totir_1, totir_0)

def mini_ModuleDefs__get_real__decision__irramt__1(irramt_0: Real, irramt_1: Real, IF_0_12: bool):
    return np.where(IF_0_12, irramt_1, irramt_0)

def mini_ModuleDefs__get_real__decision__IF_0__19(IF_0_0: bool, IF_0_1: bool, IF_0_12: bool):
    return np.where(IF_0_12, IF_0_1, IF_0_0)

def mini_ModuleDefs__get_real__decision__depir__1(depir_0: Array, depir_1: Array, IF_0_12: bool):
    return np.where(IF_0_12, depir_1, depir_0)

def mini_ModuleDefs__get_real__decision__toteffirr__1(toteffirr_0: Real, toteffirr_1: Real, IF_0_12: bool):
    return np.where(IF_0_12, toteffirr_1, toteffirr_0)

def mini_ModuleDefs__get_real__condition__IF_0__20(varname):
    return (varname == "water")

def mini_ModuleDefs__get_real__assign__wat_tmp___1(water: wattype):
    return water

def mini_ModuleDefs__get_real__condition__IF_0__21(varname):
    return (varname == "drain")

def mini_ModuleDefs__get_real__assign__value__80(drain: Real):
    return drain

def mini_ModuleDefs__get_real__decision__value__81(value_0: Real, value_1: Real, IF_0_21: bool):
    return np.where(IF_0_21, value_1, value_0)

def mini_ModuleDefs__get_real__decision__drain__0(drain_0: Real, drain_1: Real, IF_0_21: bool):
    return np.where(IF_0_21, drain_1, drain_0)

def mini_ModuleDefs__get_real__condition__IF_0__22(varname):
    return (varname == "runoff")

def mini_ModuleDefs__get_real__assign__value__82(runoff: Real):
    return runoff

def mini_ModuleDefs__get_real__decision__value__83(value_0: Real, value_1: Real, IF_0_22: bool):
    return np.where(IF_0_22, value_1, value_0)

def mini_ModuleDefs__get_real__decision__runoff__0(runoff_0: int, runoff_1: int, IF_0_22: bool):
    return np.where(IF_0_22, runoff_1, runoff_0)

def mini_ModuleDefs__get_real__condition__IF_0__23(varname):
    return (varname == "snow")

def mini_ModuleDefs__get_real__assign__value__84(snow: Real):
    return snow

def mini_ModuleDefs__get_real__assign__err__10():
    return True

def mini_ModuleDefs__get_real__decision__value__85(value_0: Real, value_1: Real, IF_0_23: bool):
    return np.where(IF_0_23, value_1, value_0)

def mini_ModuleDefs__get_real__decision__err__11(err_0: bool, err_1: bool, IF_0_23: bool):
    return np.where(IF_0_23, err_1, err_0)

def mini_ModuleDefs__get_real__decision__snow__0(snow_0: int, snow_1: int, IF_0_23: bool):
    return np.where(IF_0_23, snow_1, snow_0)

def mini_ModuleDefs__get_real__decision__value__86(value_0: Real, value_1: Real, IF_0_20: bool):
    return np.where(IF_0_20, value_1, value_0)

def mini_ModuleDefs__get_real__decision__drain__1(drain_0: Real, drain_1: Real, IF_0_20: bool):
    return np.where(IF_0_20, drain_1, drain_0)

def mini_ModuleDefs__get_real__decision__err__12(err_0: bool, err_1: bool, IF_0_20: bool):
    return np.where(IF_0_20, err_1, err_0)

def mini_ModuleDefs__get_real__decision__water__0(water_0: wattype, water_1: wattype, IF_0_20: bool):
    return np.where(IF_0_20, water_1, water_0)

def mini_ModuleDefs__get_real__decision__snow__1(snow_0: int, snow_1: int, IF_0_20: bool):
    return np.where(IF_0_20, snow_1, snow_0)

def mini_ModuleDefs__get_real__decision__IF_0__24(IF_0_0: bool, IF_0_1: bool, IF_0_20: bool):
    return np.where(IF_0_20, IF_0_1, IF_0_0)

def mini_ModuleDefs__get_real__decision__runoff__1(runoff_0: int, runoff_1: int, IF_0_20: bool):
    return np.where(IF_0_20, runoff_1, runoff_0)

def mini_ModuleDefs__get_real__condition__IF_0__25(varname):
    return (varname == "nitr")

def mini_ModuleDefs__get_real__assign__ni_tmp___1(nitr: nitype):
    return nitr

def mini_ModuleDefs__get_real__condition__IF_0__26(varname):
    return (varname == "tnoxd")

def mini_ModuleDefs__get_real__assign__value__87(tnoxd: Real):
    return tnoxd

def mini_ModuleDefs__get_real__decision__value__88(value_0: Real, value_1: Real, IF_0_26: bool):
    return np.where(IF_0_26, value_1, value_0)

def mini_ModuleDefs__get_real__decision__tnoxd__0(tnoxd_0: int, tnoxd_1: int, IF_0_26: bool):
    return np.where(IF_0_26, tnoxd_1, tnoxd_0)

def mini_ModuleDefs__get_real__condition__IF_0__27(varname):
    return (varname == "tlchd")

def mini_ModuleDefs__get_real__assign__value__89(tleachd: Real):
    return tleachd

def mini_ModuleDefs__get_real__assign__err__13():
    return True

def mini_ModuleDefs__get_real__decision__value__90(value_0: Real, value_1: Real, IF_0_27: bool):
    return np.where(IF_0_27, value_1, value_0)

def mini_ModuleDefs__get_real__decision__err__14(err_0: bool, err_1: bool, IF_0_27: bool):
    return np.where(IF_0_27, err_1, err_0)

def mini_ModuleDefs__get_real__decision__tleachd__0(tleachd_0: Real, tleachd_1: Real, IF_0_27: bool):
    return np.where(IF_0_27, tleachd_1, tleachd_0)

def mini_ModuleDefs__get_real__decision__value__91(value_0: Real, value_1: Real, IF_0_25: bool):
    return np.where(IF_0_25, value_1, value_0)

def mini_ModuleDefs__get_real__decision__err__15(err_0: bool, err_1: bool, IF_0_25: bool):
    return np.where(IF_0_25, err_1, err_0)

def mini_ModuleDefs__get_real__decision__nitr__0(nitr_0: int, nitr_1: int, IF_0_25: bool):
    return np.where(IF_0_25, nitr_1, nitr_0)

def mini_ModuleDefs__get_real__decision__tnoxd__1(tnoxd_0: int, tnoxd_1: int, IF_0_25: bool):
    return np.where(IF_0_25, tnoxd_1, tnoxd_0)

def mini_ModuleDefs__get_real__decision__tleachd__1(tleachd_0: Real, tleachd_1: Real, IF_0_25: bool):
    return np.where(IF_0_25, tleachd_1, tleachd_0)

def mini_ModuleDefs__get_real__decision__IF_0__28(IF_0_0: bool, IF_0_1: bool, IF_0_25: bool):
    return np.where(IF_0_25, IF_0_1, IF_0_0)

def mini_ModuleDefs__get_real__condition__IF_0__29(varname):
    return (varname == "orgc")

def mini_ModuleDefs__get_real__assign__orgc_tmp___1(orgc: orgctype):
    return orgc

def mini_ModuleDefs__get_real__condition__IF_0__30(varname):
    return (varname == "mulchmass")

def mini_ModuleDefs__get_real__assign__value__92(mulchmass: Real):
    return mulchmass

def mini_ModuleDefs__get_real__decision__value__93(value_0: Real, value_1: Real, IF_0_30: bool):
    return np.where(IF_0_30, value_1, value_0)

def mini_ModuleDefs__get_real__decision__mulchmass__0(mulchmass_0: Real, mulchmass_1: Real, IF_0_30: bool):
    return np.where(IF_0_30, mulchmass_1, mulchmass_0)

def mini_ModuleDefs__get_real__condition__IF_0__31(varname):
    return (varname == "tominfom")

def mini_ModuleDefs__get_real__assign__value__94(tominfom: Real):
    return tominfom

def mini_ModuleDefs__get_real__decision__value__95(value_0: Real, value_1: Real, IF_0_31: bool):
    return np.where(IF_0_31, value_1, value_0)

def mini_ModuleDefs__get_real__decision__tominfom__0(tominfom_0: int, tominfom_1: int, IF_0_31: bool):
    return np.where(IF_0_31, tominfom_1, tominfom_0)

def mini_ModuleDefs__get_real__condition__IF_0__32(varname):
    return (varname == "tominsom")

def mini_ModuleDefs__get_real__assign__value__96(tominsom: Real):
    return tominsom

def mini_ModuleDefs__get_real__decision__value__97(value_0: Real, value_1: Real, IF_0_32: bool):
    return np.where(IF_0_32, value_1, value_0)

def mini_ModuleDefs__get_real__decision__tominsom__0(tominsom_0: int, tominsom_1: int, IF_0_32: bool):
    return np.where(IF_0_32, tominsom_1, tominsom_0)

def mini_ModuleDefs__get_real__condition__IF_0__33(varname):
    return (varname == "tominsom1")

def mini_ModuleDefs__get_real__assign__value__98(tominsom1: Real):
    return tominsom1

def mini_ModuleDefs__get_real__decision__value__99(value_0: Real, value_1: Real, IF_0_33: bool):
    return np.where(IF_0_33, value_1, value_0)

def mini_ModuleDefs__get_real__decision__tominsom1__0(tominsom1_0: int, tominsom1_1: int, IF_0_33: bool):
    return np.where(IF_0_33, tominsom1_1, tominsom1_0)

def mini_ModuleDefs__get_real__condition__IF_0__34(varname):
    return (varname == "tominsom2")

def mini_ModuleDefs__get_real__assign__value__100(tominsom2: Real):
    return tominsom2

def mini_ModuleDefs__get_real__decision__value__101(value_0: Real, value_1: Real, IF_0_34: bool):
    return np.where(IF_0_34, value_1, value_0)

def mini_ModuleDefs__get_real__decision__tominsom2__0(tominsom2_0: int, tominsom2_1: int, IF_0_34: bool):
    return np.where(IF_0_34, tominsom2_1, tominsom2_0)

def mini_ModuleDefs__get_real__condition__IF_0__35(varname):
    return (varname == "tominsom3")

def mini_ModuleDefs__get_real__assign__value__102(tominsom3: Real):
    return tominsom3

def mini_ModuleDefs__get_real__decision__value__103(value_0: Real, value_1: Real, IF_0_35: bool):
    return np.where(IF_0_35, value_1, value_0)

def mini_ModuleDefs__get_real__decision__tominsom3__0(tominsom3_0: int, tominsom3_1: int, IF_0_35: bool):
    return np.where(IF_0_35, tominsom3_1, tominsom3_0)

def mini_ModuleDefs__get_real__condition__IF_0__36(varname):
    return (varname == "tnimbsom")

def mini_ModuleDefs__get_real__assign__value__104(tnimbsom: Real):
    return tnimbsom

def mini_ModuleDefs__get_real__assign__err__16():
    return True

def mini_ModuleDefs__get_real__decision__value__105(value_0: Real, value_1: Real, IF_0_36: bool):
    return np.where(IF_0_36, value_1, value_0)

def mini_ModuleDefs__get_real__decision__err__17(err_0: bool, err_1: bool, IF_0_36: bool):
    return np.where(IF_0_36, err_1, err_0)

def mini_ModuleDefs__get_real__decision__tnimbsom__0(tnimbsom_0: int, tnimbsom_1: int, IF_0_36: bool):
    return np.where(IF_0_36, tnimbsom_1, tnimbsom_0)

def mini_ModuleDefs__get_real__decision__value__106(value_0: Real, value_1: Real, IF_0_29: bool):
    return np.where(IF_0_29, value_1, value_0)

def mini_ModuleDefs__get_real__decision__tominfom__1(tominfom_0: int, tominfom_1: int, IF_0_29: bool):
    return np.where(IF_0_29, tominfom_1, tominfom_0)

def mini_ModuleDefs__get_real__decision__tominsom3__1(tominsom3_0: int, tominsom3_1: int, IF_0_29: bool):
    return np.where(IF_0_29, tominsom3_1, tominsom3_0)

def mini_ModuleDefs__get_real__decision__tominsom2__1(tominsom2_0: int, tominsom2_1: int, IF_0_29: bool):
    return np.where(IF_0_29, tominsom2_1, tominsom2_0)

def mini_ModuleDefs__get_real__decision__orgc__0(orgc_0: orgctype, orgc_1: orgctype, IF_0_29: bool):
    return np.where(IF_0_29, orgc_1, orgc_0)

def mini_ModuleDefs__get_real__decision__tominsom1__1(tominsom1_0: int, tominsom1_1: int, IF_0_29: bool):
    return np.where(IF_0_29, tominsom1_1, tominsom1_0)

def mini_ModuleDefs__get_real__decision__mulchmass__1(mulchmass_0: Real, mulchmass_1: Real, IF_0_29: bool):
    return np.where(IF_0_29, mulchmass_1, mulchmass_0)

def mini_ModuleDefs__get_real__decision__err__18(err_0: bool, err_1: bool, IF_0_29: bool):
    return np.where(IF_0_29, err_1, err_0)

def mini_ModuleDefs__get_real__decision__IF_0__37(IF_0_0: bool, IF_0_1: bool, IF_0_29: bool):
    return np.where(IF_0_29, IF_0_1, IF_0_0)

def mini_ModuleDefs__get_real__decision__tominsom__1(tominsom_0: int, tominsom_1: int, IF_0_29: bool):
    return np.where(IF_0_29, tominsom_1, tominsom_0)

def mini_ModuleDefs__get_real__decision__tnimbsom__1(tnimbsom_0: int, tnimbsom_1: int, IF_0_29: bool):
    return np.where(IF_0_29, tnimbsom_1, tnimbsom_0)

def mini_ModuleDefs__get_real__condition__IF_0__38(varname):
    return (varname == "soil")

def mini_ModuleDefs__get_real__assign__orgc_tmp___1(orgc: orgctype):
    return orgc

def mini_ModuleDefs__get_real__condition__IF_0__39(varname):
    return (varname == "tominfom")

def mini_ModuleDefs__get_real__assign__value__107(tominfom: Real):
    return tominfom

def mini_ModuleDefs__get_real__decision__value__108(value_0: Real, value_1: Real, IF_0_39: bool):
    return np.where(IF_0_39, value_1, value_0)

def mini_ModuleDefs__get_real__condition__IF_0__40(varname):
    return (varname == "tominsom")

def mini_ModuleDefs__get_real__assign__value__109(tominsom: Real):
    return tominsom

def mini_ModuleDefs__get_real__decision__value__110(value_0: Real, value_1: Real, IF_0_40: bool):
    return np.where(IF_0_40, value_1, value_0)

def mini_ModuleDefs__get_real__condition__IF_0__41(varname):
    return (varname == "tominsom1")

def mini_ModuleDefs__get_real__assign__value__111(tominsom1: Real):
    return tominsom1

def mini_ModuleDefs__get_real__decision__value__112(value_0: Real, value_1: Real, IF_0_41: bool):
    return np.where(IF_0_41, value_1, value_0)

def mini_ModuleDefs__get_real__condition__IF_0__42(varname):
    return (varname == "tominsom2")

def mini_ModuleDefs__get_real__assign__value__113(tominsom2: Real):
    return tominsom2

def mini_ModuleDefs__get_real__decision__value__114(value_0: Real, value_1: Real, IF_0_42: bool):
    return np.where(IF_0_42, value_1, value_0)

def mini_ModuleDefs__get_real__condition__IF_0__43(varname):
    return (varname == "tominsom3")

def mini_ModuleDefs__get_real__assign__value__115(tominsom3: Real):
    return tominsom3

def mini_ModuleDefs__get_real__decision__value__116(value_0: Real, value_1: Real, IF_0_43: bool):
    return np.where(IF_0_43, value_1, value_0)

def mini_ModuleDefs__get_real__condition__IF_0__44(varname):
    return (varname == "tnimbsom")

def mini_ModuleDefs__get_real__assign__value__117(tnimbsom: Real):
    return tnimbsom

def mini_ModuleDefs__get_real__assign__err__19():
    return True

def mini_ModuleDefs__get_real__decision__value__118(value_0: Real, value_1: Real, IF_0_44: bool):
    return np.where(IF_0_44, value_1, value_0)

def mini_ModuleDefs__get_real__decision__err__20(err_0: bool, err_1: bool, IF_0_44: bool):
    return np.where(IF_0_44, err_1, err_0)

def mini_ModuleDefs__get_real__decision__value__119(value_0: Real, value_1: Real, IF_0_38: bool):
    return np.where(IF_0_38, value_1, value_0)

def mini_ModuleDefs__get_real__decision__err__21(err_0: bool, err_1: bool, IF_0_38: bool):
    return np.where(IF_0_38, err_1, err_0)

def mini_ModuleDefs__get_real__decision__IF_0__45(IF_0_0: bool, IF_0_1: bool, IF_0_38: bool):
    return np.where(IF_0_38, IF_0_1, IF_0_0)

def mini_ModuleDefs__get_real__condition__IF_0__46(varname):
    return (varname == "pdlabeta")

def mini_ModuleDefs__get_real__assign__pdlabeta_tmp___1(pdlabeta: pdlabetatype):
    return pdlabeta

def mini_ModuleDefs__get_real__condition__IF_0__47(varname):
    return (varname == "pdla")

def mini_ModuleDefs__get_real__assign__value__120(pdla: Real):
    return pdla

def mini_ModuleDefs__get_real__decision__value__121(value_0: Real, value_1: Real, IF_0_47: bool):
    return np.where(IF_0_47, value_1, value_0)

def mini_ModuleDefs__get_real__decision__pdla__0(pdla_0: int, pdla_1: int, IF_0_47: bool):
    return np.where(IF_0_47, pdla_1, pdla_0)

def mini_ModuleDefs__get_real__condition__IF_0__48(varname):
    return (varname == "beta")

def mini_ModuleDefs__get_real__assign__value__122(betals: Real):
    return betals

def mini_ModuleDefs__get_real__assign__err__22():
    return True

def mini_ModuleDefs__get_real__decision__value__123(value_0: Real, value_1: Real, IF_0_48: bool):
    return np.where(IF_0_48, value_1, value_0)

def mini_ModuleDefs__get_real__decision__betals__0(betals_0: Real, betals_1: Real, IF_0_48: bool):
    return np.where(IF_0_48, betals_1, betals_0)

def mini_ModuleDefs__get_real__decision__err__23(err_0: bool, err_1: bool, IF_0_48: bool):
    return np.where(IF_0_48, err_1, err_0)

def mini_ModuleDefs__get_real__assign__err__24():
    return True

def mini_ModuleDefs__get_real__decision__value__124(value_0: Real, value_1: Real, IF_0_46: bool):
    return np.where(IF_0_46, value_1, value_0)

def mini_ModuleDefs__get_real__decision__betals__1(betals_0: Real, betals_1: Real, IF_0_46: bool):
    return np.where(IF_0_46, betals_1, betals_0)

def mini_ModuleDefs__get_real__decision__pdlabeta__0(pdlabeta_0: Real, pdlabeta_1: Real, IF_0_46: bool):
    return np.where(IF_0_46, pdlabeta_1, pdlabeta_0)

def mini_ModuleDefs__get_real__decision__err__25(err_0: bool, err_1: bool, IF_0_46: bool):
    return np.where(IF_0_46, err_1, err_0)

def mini_ModuleDefs__get_real__decision__pdla__1(pdla_0: int, pdla_1: int, IF_0_46: bool):
    return np.where(IF_0_46, pdla_1, pdla_0)

def mini_ModuleDefs__get_real__decision__IF_0__49(IF_0_0: bool, IF_0_1: bool, IF_0_46: bool):
    return np.where(IF_0_46, IF_0_1, IF_0_0)

def mini_ModuleDefs__put_real__assign__err__0():
    return False

def mini_ModuleDefs__put_real__condition__IF_0__0(modulename):
    return (modulename == "spam")

def mini_ModuleDefs__put_real__assign__spam_tmp___1(spam: spamtype):
    return spam

def mini_ModuleDefs__put_real__condition__IF_0__1(varname):
    return (varname == "agefac")

def mini_ModuleDefs__put_real__assign__spam_tmp__agefac___1(value: Real, spam_tmp: spamtype):
    spam_tmp.agefac = value
    return spam_tmp.agefac

def mini_ModuleDefs__put_real__decision__spam_tmp__agefac__0(spam_tmp__agefac_0: int, spam_tmp__agefac_1: int, IF_0_1: bool):
    return np.where(IF_0_1, spam_tmp__agefac_1, spam_tmp__agefac_0)

def mini_ModuleDefs__put_real__decision__agefac__0(agefac_0: Real, agefac_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, agefac_1, agefac_0)

def mini_ModuleDefs__put_real__condition__IF_0__2(varname):
    return (varname == "pg")

def mini_ModuleDefs__put_real__assign__spam_tmp__pg___1(value: Real, spam_tmp: spamtype):
    spam_tmp.pg = value
    return spam_tmp.pg

def mini_ModuleDefs__put_real__decision__pg__0(pg_0: int, pg_1: int, IF_0_2: bool):
    return np.where(IF_0_2, pg_1, pg_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__pg__0(spam_tmp__pg_0: int, spam_tmp__pg_1: int, IF_0_2: bool):
    return np.where(IF_0_2, spam_tmp__pg_1, spam_tmp__pg_0)

def mini_ModuleDefs__put_real__condition__IF_0__3(varname):
    return (varname == "cef")

def mini_ModuleDefs__put_real__assign__spam_tmp__cef___1(value: Real, spam_tmp: spamtype):
    spam_tmp.cef = value
    return spam_tmp.cef

def mini_ModuleDefs__put_real__decision__cef__0(cef_0: Real, cef_1: Real, IF_0_3: bool):
    return np.where(IF_0_3, cef_1, cef_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__cef__0(spam_tmp__cef_0: int, spam_tmp__cef_1: int, IF_0_3: bool):
    return np.where(IF_0_3, spam_tmp__cef_1, spam_tmp__cef_0)

def mini_ModuleDefs__put_real__condition__IF_0__4(varname):
    return (varname == "cem")

def mini_ModuleDefs__put_real__assign__spam_tmp__cem___1(value: Real, spam_tmp: spamtype):
    spam_tmp.cem = value
    return spam_tmp.cem

def mini_ModuleDefs__put_real__decision__spam_tmp__cem__0(spam_tmp__cem_0: int, spam_tmp__cem_1: int, IF_0_4: bool):
    return np.where(IF_0_4, spam_tmp__cem_1, spam_tmp__cem_0)

def mini_ModuleDefs__put_real__decision__cem__0(cem_0: Real, cem_1: Real, IF_0_4: bool):
    return np.where(IF_0_4, cem_1, cem_0)

def mini_ModuleDefs__put_real__condition__IF_0__5(varname):
    return (varname == "ceo")

def mini_ModuleDefs__put_real__assign__spam_tmp__ceo___1(value: Real, spam_tmp: spamtype):
    spam_tmp.ceo = value
    return spam_tmp.ceo

def mini_ModuleDefs__put_real__decision__spam_tmp__ceo__0(spam_tmp__ceo_0: int, spam_tmp__ceo_1: int, IF_0_5: bool):
    return np.where(IF_0_5, spam_tmp__ceo_1, spam_tmp__ceo_0)

def mini_ModuleDefs__put_real__decision__ceo__0(ceo_0: Real, ceo_1: Real, IF_0_5: bool):
    return np.where(IF_0_5, ceo_1, ceo_0)

def mini_ModuleDefs__put_real__condition__IF_0__6(varname):
    return (varname == "cep")

def mini_ModuleDefs__put_real__assign__spam_tmp__cep___1(value: Real, spam_tmp: spamtype):
    spam_tmp.cep = value
    return spam_tmp.cep

def mini_ModuleDefs__put_real__decision__cep__0(cep_0: int, cep_1: int, IF_0_6: bool):
    return np.where(IF_0_6, cep_1, cep_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__cep__0(spam_tmp__cep_0: int, spam_tmp__cep_1: int, IF_0_6: bool):
    return np.where(IF_0_6, spam_tmp__cep_1, spam_tmp__cep_0)

def mini_ModuleDefs__put_real__condition__IF_0__7(varname):
    return (varname == "ces")

def mini_ModuleDefs__put_real__assign__spam_tmp__ces___1(value: Real, spam_tmp: spamtype):
    spam_tmp.ces = value
    return spam_tmp.ces

def mini_ModuleDefs__put_real__decision__ces__0(ces_0: Real, ces_1: Real, IF_0_7: bool):
    return np.where(IF_0_7, ces_1, ces_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__ces__0(spam_tmp__ces_0: int, spam_tmp__ces_1: int, IF_0_7: bool):
    return np.where(IF_0_7, spam_tmp__ces_1, spam_tmp__ces_0)

def mini_ModuleDefs__put_real__condition__IF_0__8(varname):
    return (varname == "cet")

def mini_ModuleDefs__put_real__assign__spam_tmp__cet___1(value: Real, spam_tmp: spamtype):
    spam_tmp.cet = value
    return spam_tmp.cet

def mini_ModuleDefs__put_real__decision__cet__0(cet_0: Real, cet_1: Real, IF_0_8: bool):
    return np.where(IF_0_8, cet_1, cet_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__cet__0(spam_tmp__cet_0: int, spam_tmp__cet_1: int, IF_0_8: bool):
    return np.where(IF_0_8, spam_tmp__cet_1, spam_tmp__cet_0)

def mini_ModuleDefs__put_real__condition__IF_0__9(varname):
    return (varname == "ef")

def mini_ModuleDefs__put_real__assign__spam_tmp__ef___1(value: Real, spam_tmp: spamtype):
    spam_tmp.ef = value
    return spam_tmp.ef

def mini_ModuleDefs__put_real__decision__spam_tmp__ef__0(spam_tmp__ef_0: int, spam_tmp__ef_1: int, IF_0_9: bool):
    return np.where(IF_0_9, spam_tmp__ef_1, spam_tmp__ef_0)

def mini_ModuleDefs__put_real__decision__ef__0(ef_0: Real, ef_1: Real, IF_0_9: bool):
    return np.where(IF_0_9, ef_1, ef_0)

def mini_ModuleDefs__put_real__condition__IF_0__10(varname):
    return (varname == "em")

def mini_ModuleDefs__put_real__assign__spam_tmp__em___1(value: Real, spam_tmp: spamtype):
    spam_tmp.em = value
    return spam_tmp.em

def mini_ModuleDefs__put_real__decision__spam_tmp__em__0(spam_tmp__em_0: int, spam_tmp__em_1: int, IF_0_10: bool):
    return np.where(IF_0_10, spam_tmp__em_1, spam_tmp__em_0)

def mini_ModuleDefs__put_real__decision__em__0(em_0: Real, em_1: Real, IF_0_10: bool):
    return np.where(IF_0_10, em_1, em_0)

def mini_ModuleDefs__put_real__condition__IF_0__11(varname):
    return (varname == "eo")

def mini_ModuleDefs__put_real__assign__spam_tmp__eo___1(value: Real, spam_tmp: spamtype):
    spam_tmp.eo = value
    return spam_tmp.eo

def mini_ModuleDefs__put_real__decision__spam_tmp__eo__0(spam_tmp__eo_0: int, spam_tmp__eo_1: int, IF_0_11: bool):
    return np.where(IF_0_11, spam_tmp__eo_1, spam_tmp__eo_0)

def mini_ModuleDefs__put_real__decision__eo__0(eo_0: Real, eo_1: Real, IF_0_11: bool):
    return np.where(IF_0_11, eo_1, eo_0)

def mini_ModuleDefs__put_real__condition__IF_0__12(varname):
    return (varname == "ep")

def mini_ModuleDefs__put_real__assign__spam_tmp__ep___1(value: Real, spam_tmp: spamtype):
    spam_tmp.ep = value
    return spam_tmp.ep

def mini_ModuleDefs__put_real__decision__spam_tmp__ep__0(spam_tmp__ep_0: int, spam_tmp__ep_1: int, IF_0_12: bool):
    return np.where(IF_0_12, spam_tmp__ep_1, spam_tmp__ep_0)

def mini_ModuleDefs__put_real__decision__ep__0(ep_0: int, ep_1: int, IF_0_12: bool):
    return np.where(IF_0_12, ep_1, ep_0)

def mini_ModuleDefs__put_real__condition__IF_0__13(varname):
    return (varname == "es")

def mini_ModuleDefs__put_real__assign__spam_tmp__es___1(value: Real, spam_tmp: spamtype):
    spam_tmp.es = value
    return spam_tmp.es

def mini_ModuleDefs__put_real__decision__spam_tmp__es__0(spam_tmp__es_0: int, spam_tmp__es_1: int, IF_0_13: bool):
    return np.where(IF_0_13, spam_tmp__es_1, spam_tmp__es_0)

def mini_ModuleDefs__put_real__decision__es__0(es_0: Real, es_1: Real, IF_0_13: bool):
    return np.where(IF_0_13, es_1, es_0)

def mini_ModuleDefs__put_real__condition__IF_0__14(varname):
    return (varname == "et")

def mini_ModuleDefs__put_real__assign__spam_tmp__et___1(value: Real, spam_tmp: spamtype):
    spam_tmp.et = value
    return spam_tmp.et

def mini_ModuleDefs__put_real__decision__spam_tmp__et__0(spam_tmp__et_0: int, spam_tmp__et_1: int, IF_0_14: bool):
    return np.where(IF_0_14, spam_tmp__et_1, spam_tmp__et_0)

def mini_ModuleDefs__put_real__decision__et__0(et_0: Real, et_1: Real, IF_0_14: bool):
    return np.where(IF_0_14, et_1, et_0)

def mini_ModuleDefs__put_real__condition__IF_0__15(varname):
    return (varname == "eop")

def mini_ModuleDefs__put_real__assign__spam_tmp__eop___1(value: Real, spam_tmp: spamtype):
    spam_tmp.eop = value
    return spam_tmp.eop

def mini_ModuleDefs__put_real__decision__eop__0(eop_0: int, eop_1: int, IF_0_15: bool):
    return np.where(IF_0_15, eop_1, eop_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__eop__0(spam_tmp__eop_0: int, spam_tmp__eop_1: int, IF_0_15: bool):
    return np.where(IF_0_15, spam_tmp__eop_1, spam_tmp__eop_0)

def mini_ModuleDefs__put_real__condition__IF_0__16(varname):
    return (varname == "evap")

def mini_ModuleDefs__put_real__assign__spam_tmp__evap___1(value: Real, spam_tmp: spamtype):
    spam_tmp.evap = value
    return spam_tmp.evap

def mini_ModuleDefs__put_real__decision__spam_tmp__evap__0(spam_tmp__evap_0: int, spam_tmp__evap_1: int, IF_0_16: bool):
    return np.where(IF_0_16, spam_tmp__evap_1, spam_tmp__evap_0)

def mini_ModuleDefs__put_real__decision__evap__0(evap_0: int, evap_1: int, IF_0_16: bool):
    return np.where(IF_0_16, evap_1, evap_0)

def mini_ModuleDefs__put_real__condition__IF_0__17(varname):
    return (varname == "refet")

def mini_ModuleDefs__put_real__assign__spam_tmp__refet___1(value: Real, spam_tmp: spamtype):
    spam_tmp.refet = value
    return spam_tmp.refet

def mini_ModuleDefs__put_real__decision__spam_tmp__refet__0(spam_tmp__refet_0: int, spam_tmp__refet_1: int, IF_0_17: bool):
    return np.where(IF_0_17, spam_tmp__refet_1, spam_tmp__refet_0)

def mini_ModuleDefs__put_real__decision__refet__0(refet_0: Real, refet_1: Real, IF_0_17: bool):
    return np.where(IF_0_17, refet_1, refet_0)

def mini_ModuleDefs__put_real__condition__IF_0__18(varname):
    return (varname == "skc")

def mini_ModuleDefs__put_real__assign__spam_tmp__skc___1(value: Real, spam_tmp: spamtype):
    spam_tmp.skc = value
    return spam_tmp.skc

def mini_ModuleDefs__put_real__decision__spam_tmp__skc__0(spam_tmp__skc_0: int, spam_tmp__skc_1: int, IF_0_18: bool):
    return np.where(IF_0_18, spam_tmp__skc_1, spam_tmp__skc_0)

def mini_ModuleDefs__put_real__decision__skc__0(skc_0: Real, skc_1: Real, IF_0_18: bool):
    return np.where(IF_0_18, skc_1, skc_0)

def mini_ModuleDefs__put_real__condition__IF_0__19(varname):
    return (varname == "kcbmin")

def mini_ModuleDefs__put_real__assign__spam_tmp__kcbmin___1(value: Real, spam_tmp: spamtype):
    spam_tmp.kcbmin = value
    return spam_tmp.kcbmin

def mini_ModuleDefs__put_real__decision__kcbmin__0(kcbmin_0: int, kcbmin_1: int, IF_0_19: bool):
    return np.where(IF_0_19, kcbmin_1, kcbmin_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__kcbmin__0(spam_tmp__kcbmin_0: int, spam_tmp__kcbmin_1: int, IF_0_19: bool):
    return np.where(IF_0_19, spam_tmp__kcbmin_1, spam_tmp__kcbmin_0)

def mini_ModuleDefs__put_real__condition__IF_0__20(varname):
    return (varname == "kcbmax")

def mini_ModuleDefs__put_real__assign__spam_tmp__kcbmax___1(value: Real, spam_tmp: spamtype):
    spam_tmp.kcbmax = value
    return spam_tmp.kcbmax

def mini_ModuleDefs__put_real__decision__kcbmax__0(kcbmax_0: Real, kcbmax_1: Real, IF_0_20: bool):
    return np.where(IF_0_20, kcbmax_1, kcbmax_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__kcbmax__0(spam_tmp__kcbmax_0: int, spam_tmp__kcbmax_1: int, IF_0_20: bool):
    return np.where(IF_0_20, spam_tmp__kcbmax_1, spam_tmp__kcbmax_0)

def mini_ModuleDefs__put_real__condition__IF_0__21(varname):
    return (varname == "kcb")

def mini_ModuleDefs__put_real__assign__spam_tmp__kcb___1(value: Real, spam_tmp: spamtype):
    spam_tmp.kcb = value
    return spam_tmp.kcb

def mini_ModuleDefs__put_real__decision__kcb__0(kcb_0: Real, kcb_1: Real, IF_0_21: bool):
    return np.where(IF_0_21, kcb_1, kcb_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__kcb__0(spam_tmp__kcb_0: int, spam_tmp__kcb_1: int, IF_0_21: bool):
    return np.where(IF_0_21, spam_tmp__kcb_1, spam_tmp__kcb_0)

def mini_ModuleDefs__put_real__condition__IF_0__22(varname):
    return (varname == "ke")

def mini_ModuleDefs__put_real__assign__spam_tmp__ke___1(value: Real, spam_tmp: spamtype):
    spam_tmp.ke = value
    return spam_tmp.ke

def mini_ModuleDefs__put_real__decision__spam_tmp__ke__0(spam_tmp__ke_0: int, spam_tmp__ke_1: int, IF_0_22: bool):
    return np.where(IF_0_22, spam_tmp__ke_1, spam_tmp__ke_0)

def mini_ModuleDefs__put_real__decision__ke__0(ke_0: Real, ke_1: Real, IF_0_22: bool):
    return np.where(IF_0_22, ke_1, ke_0)

def mini_ModuleDefs__put_real__condition__IF_0__23(varname):
    return (varname == "kc")

def mini_ModuleDefs__put_real__assign__spam_tmp__kc___1(value: Real, spam_tmp: spamtype):
    spam_tmp.kc = value
    return spam_tmp.kc

def mini_ModuleDefs__put_real__assign__err__1():
    return True

def mini_ModuleDefs__put_real__decision__err__2(err_0: bool, err_1: bool, IF_0_23: bool):
    return np.where(IF_0_23, err_1, err_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__kc__0(spam_tmp__kc_0: int, spam_tmp__kc_1: int, IF_0_23: bool):
    return np.where(IF_0_23, spam_tmp__kc_1, spam_tmp__kc_0)

def mini_ModuleDefs__put_real__decision__kc__0(kc_0: Real, kc_1: Real, IF_0_23: bool):
    return np.where(IF_0_23, kc_1, kc_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__refet__1(spam_tmp__refet_0: int, spam_tmp__refet_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__refet_1, spam_tmp__refet_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__skc__1(spam_tmp__skc_0: int, spam_tmp__skc_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__skc_1, spam_tmp__skc_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__cem__1(spam_tmp__cem_0: int, spam_tmp__cem_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__cem_1, spam_tmp__cem_0)

def mini_ModuleDefs__put_real__decision__cet__1(cet_0: Real, cet_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, cet_1, cet_0)

def mini_ModuleDefs__put_real__decision__eop__1(eop_0: int, eop_1: int, IF_0_0: bool):
    return np.where(IF_0_0, eop_1, eop_0)

def mini_ModuleDefs__put_real__decision__agefac__1(agefac_0: Real, agefac_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, agefac_1, agefac_0)

def mini_ModuleDefs__put_real__decision__kcbmin__1(kcbmin_0: int, kcbmin_1: int, IF_0_0: bool):
    return np.where(IF_0_0, kcbmin_1, kcbmin_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__ep__1(spam_tmp__ep_0: int, spam_tmp__ep_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__ep_1, spam_tmp__ep_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__cet__1(spam_tmp__cet_0: int, spam_tmp__cet_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__cet_1, spam_tmp__cet_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__em__1(spam_tmp__em_0: int, spam_tmp__em_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__em_1, spam_tmp__em_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__evap__1(spam_tmp__evap_0: int, spam_tmp__evap_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__evap_1, spam_tmp__evap_0)

def mini_ModuleDefs__put_real__decision__err__3(err_0: bool, err_1: bool, IF_0_0: bool):
    return np.where(IF_0_0, err_1, err_0)

def mini_ModuleDefs__put_real__decision__ceo__1(ceo_0: Real, ceo_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, ceo_1, ceo_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__ke__1(spam_tmp__ke_0: int, spam_tmp__ke_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__ke_1, spam_tmp__ke_0)

def mini_ModuleDefs__put_real__decision__ke__1(ke_0: Real, ke_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, ke_1, ke_0)

def mini_ModuleDefs__put_real__decision__ep__1(ep_0: int, ep_1: int, IF_0_0: bool):
    return np.where(IF_0_0, ep_1, ep_0)

def mini_ModuleDefs__put_real__decision__kcbmax__1(kcbmax_0: Real, kcbmax_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, kcbmax_1, kcbmax_0)

def mini_ModuleDefs__put_real__decision__cep__1(cep_0: int, cep_1: int, IF_0_0: bool):
    return np.where(IF_0_0, cep_1, cep_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__es__1(spam_tmp__es_0: int, spam_tmp__es_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__es_1, spam_tmp__es_0)

def mini_ModuleDefs__put_real__decision__spam__0(spam_0: int, spam_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_1, spam_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__ef__1(spam_tmp__ef_0: int, spam_tmp__ef_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__ef_1, spam_tmp__ef_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__cep__1(spam_tmp__cep_0: int, spam_tmp__cep_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__cep_1, spam_tmp__cep_0)

def mini_ModuleDefs__put_real__decision__pg__1(pg_0: int, pg_1: int, IF_0_0: bool):
    return np.where(IF_0_0, pg_1, pg_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__kc__1(spam_tmp__kc_0: int, spam_tmp__kc_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__kc_1, spam_tmp__kc_0)

def mini_ModuleDefs__put_real__decision__ef__1(ef_0: Real, ef_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, ef_1, ef_0)

def mini_ModuleDefs__put_real__decision__cem__1(cem_0: Real, cem_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, cem_1, cem_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__et__1(spam_tmp__et_0: int, spam_tmp__et_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__et_1, spam_tmp__et_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__ceo__1(spam_tmp__ceo_0: int, spam_tmp__ceo_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__ceo_1, spam_tmp__ceo_0)

def mini_ModuleDefs__put_real__decision__cef__1(cef_0: Real, cef_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, cef_1, cef_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__eo__1(spam_tmp__eo_0: int, spam_tmp__eo_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__eo_1, spam_tmp__eo_0)

def mini_ModuleDefs__put_real__decision__em__1(em_0: Real, em_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, em_1, em_0)

def mini_ModuleDefs__put_real__decision__eo__1(eo_0: Real, eo_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, eo_1, eo_0)

def mini_ModuleDefs__put_real__decision__evap__1(evap_0: int, evap_1: int, IF_0_0: bool):
    return np.where(IF_0_0, evap_1, evap_0)

def mini_ModuleDefs__put_real__decision__es__1(es_0: Real, es_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, es_1, es_0)

def mini_ModuleDefs__put_real__decision__refet__1(refet_0: Real, refet_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, refet_1, refet_0)

def mini_ModuleDefs__put_real__decision__IF_0__24(IF_0_0: bool, IF_0_1: bool):
    return np.where(IF_0_0, IF_0_1, IF_0_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__kcbmax__1(spam_tmp__kcbmax_0: int, spam_tmp__kcbmax_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__kcbmax_1, spam_tmp__kcbmax_0)

def mini_ModuleDefs__put_real__decision__kcb__1(kcb_0: Real, kcb_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, kcb_1, kcb_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__cef__1(spam_tmp__cef_0: int, spam_tmp__cef_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__cef_1, spam_tmp__cef_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__agefac__1(spam_tmp__agefac_0: int, spam_tmp__agefac_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__agefac_1, spam_tmp__agefac_0)

def mini_ModuleDefs__put_real__decision__et__1(et_0: Real, et_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, et_1, et_0)

def mini_ModuleDefs__put_real__decision__skc__1(skc_0: Real, skc_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, skc_1, skc_0)

def mini_ModuleDefs__put_real__decision__ces__1(ces_0: Real, ces_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, ces_1, ces_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__kcb__1(spam_tmp__kcb_0: int, spam_tmp__kcb_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__kcb_1, spam_tmp__kcb_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__eop__1(spam_tmp__eop_0: int, spam_tmp__eop_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__eop_1, spam_tmp__eop_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__pg__1(spam_tmp__pg_0: int, spam_tmp__pg_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__pg_1, spam_tmp__pg_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__kcbmin__1(spam_tmp__kcbmin_0: int, spam_tmp__kcbmin_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__kcbmin_1, spam_tmp__kcbmin_0)

def mini_ModuleDefs__put_real__decision__spam_tmp__ces__1(spam_tmp__ces_0: int, spam_tmp__ces_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_tmp__ces_1, spam_tmp__ces_0)

def mini_ModuleDefs__put_real__decision__kc__1(kc_0: Real, kc_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, kc_1, kc_0)

def mini_ModuleDefs__put_real__condition__IF_0__25(varname):
    return (varname == "plant")

def mini_ModuleDefs__put_real__assign__plant_tmp___1(plant: planttype):
    return plant

def mini_ModuleDefs__put_real__condition__IF_0__26(varname):
    return (varname == "biomas")

def mini_ModuleDefs__put_real__assign__plant_tmp__biomas___1(value: Real, plant_tmp: planttype):
    plant_tmp.biomas = value
    return plant_tmp.biomas

def mini_ModuleDefs__put_real__decision__biomas__0(biomas_0: Real, biomas_1: Real, IF_0_26: bool):
    return np.where(IF_0_26, biomas_1, biomas_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__biomas__0(plant_tmp__biomas_0: int, plant_tmp__biomas_1: int, IF_0_26: bool):
    return np.where(IF_0_26, plant_tmp__biomas_1, plant_tmp__biomas_0)

def mini_ModuleDefs__put_real__condition__IF_0__27(varname):
    return (varname == "canht")

def mini_ModuleDefs__put_real__assign__plant_tmp__canht___1(value: Real, plant_tmp: planttype):
    plant_tmp.canht = value
    return plant_tmp.canht

def mini_ModuleDefs__put_real__decision__plant_tmp__canht__0(plant_tmp__canht_0: int, plant_tmp__canht_1: int, IF_0_27: bool):
    return np.where(IF_0_27, plant_tmp__canht_1, plant_tmp__canht_0)

def mini_ModuleDefs__put_real__decision__canht__0(canht_0: int, canht_1: int, IF_0_27: bool):
    return np.where(IF_0_27, canht_1, canht_0)

def mini_ModuleDefs__put_real__condition__IF_0__28(varname):
    return (varname == "canwh")

def mini_ModuleDefs__put_real__assign__plant_tmp__canwh___1(value: Real, plant_tmp: planttype):
    plant_tmp.canwh = value
    return plant_tmp.canwh

def mini_ModuleDefs__put_real__decision__plant_tmp__canwh__0(plant_tmp__canwh_0: int, plant_tmp__canwh_1: int, IF_0_28: bool):
    return np.where(IF_0_28, plant_tmp__canwh_1, plant_tmp__canwh_0)

def mini_ModuleDefs__put_real__decision__canwh__0(canwh_0: int, canwh_1: int, IF_0_28: bool):
    return np.where(IF_0_28, canwh_1, canwh_0)

def mini_ModuleDefs__put_real__condition__IF_0__29(varname):
    return (varname == "dxr57")

def mini_ModuleDefs__put_real__assign__plant_tmp__dxr57___1(value: Real, plant_tmp: planttype):
    plant_tmp.dxr57 = value
    return plant_tmp.dxr57

def mini_ModuleDefs__put_real__decision__dxr57__0(dxr57_0: Real, dxr57_1: Real, IF_0_29: bool):
    return np.where(IF_0_29, dxr57_1, dxr57_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__dxr57__0(plant_tmp__dxr57_0: int, plant_tmp__dxr57_1: int, IF_0_29: bool):
    return np.where(IF_0_29, plant_tmp__dxr57_1, plant_tmp__dxr57_0)

def mini_ModuleDefs__put_real__condition__IF_0__30(varname):
    return (varname == "excess")

def mini_ModuleDefs__put_real__assign__plant_tmp__excess___1(value: Real, plant_tmp: planttype):
    plant_tmp.excess = value
    return plant_tmp.excess

def mini_ModuleDefs__put_real__decision__excess__0(excess_0: Real, excess_1: Real, IF_0_30: bool):
    return np.where(IF_0_30, excess_1, excess_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__excess__0(plant_tmp__excess_0: int, plant_tmp__excess_1: int, IF_0_30: bool):
    return np.where(IF_0_30, plant_tmp__excess_1, plant_tmp__excess_0)

def mini_ModuleDefs__put_real__condition__IF_0__31(varname):
    return (varname == "pltpop")

def mini_ModuleDefs__put_real__assign__plant_tmp__pltpop___1(value: Real, plant_tmp: planttype):
    plant_tmp.pltpop = value
    return plant_tmp.pltpop

def mini_ModuleDefs__put_real__decision__pltpop__0(pltpop_0: int, pltpop_1: int, IF_0_31: bool):
    return np.where(IF_0_31, pltpop_1, pltpop_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__pltpop__0(plant_tmp__pltpop_0: int, plant_tmp__pltpop_1: int, IF_0_31: bool):
    return np.where(IF_0_31, plant_tmp__pltpop_1, plant_tmp__pltpop_0)

def mini_ModuleDefs__put_real__condition__IF_0__32(varname):
    return (varname == "rnitp")

def mini_ModuleDefs__put_real__assign__plant_tmp__rnitp___1(value: Real, plant_tmp: planttype):
    plant_tmp.rnitp = value
    return plant_tmp.rnitp

def mini_ModuleDefs__put_real__decision__rnitp__0(rnitp_0: int, rnitp_1: int, IF_0_32: bool):
    return np.where(IF_0_32, rnitp_1, rnitp_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__rnitp__0(plant_tmp__rnitp_0: int, plant_tmp__rnitp_1: int, IF_0_32: bool):
    return np.where(IF_0_32, plant_tmp__rnitp_1, plant_tmp__rnitp_0)

def mini_ModuleDefs__put_real__condition__IF_0__33(varname):
    return (varname == "slaad")

def mini_ModuleDefs__put_real__assign__plant_tmp__slaad___1(value: Real, plant_tmp: planttype):
    plant_tmp.slaad = value
    return plant_tmp.slaad

def mini_ModuleDefs__put_real__decision__plant_tmp__slaad__0(plant_tmp__slaad_0: int, plant_tmp__slaad_1: int, IF_0_33: bool):
    return np.where(IF_0_33, plant_tmp__slaad_1, plant_tmp__slaad_0)

def mini_ModuleDefs__put_real__decision__slaad__0(slaad_0: Real, slaad_1: Real, IF_0_33: bool):
    return np.where(IF_0_33, slaad_1, slaad_0)

def mini_ModuleDefs__put_real__condition__IF_0__34(varname):
    return (varname == "xpod")

def mini_ModuleDefs__put_real__assign__plant_tmp__xpod___1(value: Real, plant_tmp: planttype):
    plant_tmp.xpod = value
    return plant_tmp.xpod

def mini_ModuleDefs__put_real__assign__err__4():
    return True

def mini_ModuleDefs__put_real__decision__xpod__0(xpod_0: int, xpod_1: int, IF_0_34: bool):
    return np.where(IF_0_34, xpod_1, xpod_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__xpod__0(plant_tmp__xpod_0: int, plant_tmp__xpod_1: int, IF_0_34: bool):
    return np.where(IF_0_34, plant_tmp__xpod_1, plant_tmp__xpod_0)

def mini_ModuleDefs__put_real__decision__err__5(err_0: bool, err_1: bool, IF_0_34: bool):
    return np.where(IF_0_34, err_1, err_0)

def mini_ModuleDefs__put_real__decision__xpod__1(xpod_0: int, xpod_1: int, IF_0_25: bool):
    return np.where(IF_0_25, xpod_1, xpod_0)

def mini_ModuleDefs__put_real__decision__plant__0(plant_0: int, plant_1: int, IF_0_25: bool):
    return np.where(IF_0_25, plant_1, plant_0)

def mini_ModuleDefs__put_real__decision__pltpop__1(pltpop_0: int, pltpop_1: int, IF_0_25: bool):
    return np.where(IF_0_25, pltpop_1, pltpop_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__canwh__1(plant_tmp__canwh_0: int, plant_tmp__canwh_1: int, IF_0_25: bool):
    return np.where(IF_0_25, plant_tmp__canwh_1, plant_tmp__canwh_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__pltpop__1(plant_tmp__pltpop_0: int, plant_tmp__pltpop_1: int, IF_0_25: bool):
    return np.where(IF_0_25, plant_tmp__pltpop_1, plant_tmp__pltpop_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__slaad__1(plant_tmp__slaad_0: int, plant_tmp__slaad_1: int, IF_0_25: bool):
    return np.where(IF_0_25, plant_tmp__slaad_1, plant_tmp__slaad_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__xpod__1(plant_tmp__xpod_0: int, plant_tmp__xpod_1: int, IF_0_25: bool):
    return np.where(IF_0_25, plant_tmp__xpod_1, plant_tmp__xpod_0)

def mini_ModuleDefs__put_real__decision__dxr57__1(dxr57_0: Real, dxr57_1: Real, IF_0_25: bool):
    return np.where(IF_0_25, dxr57_1, dxr57_0)

def mini_ModuleDefs__put_real__decision__rnitp__1(rnitp_0: int, rnitp_1: int, IF_0_25: bool):
    return np.where(IF_0_25, rnitp_1, rnitp_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__dxr57__1(plant_tmp__dxr57_0: int, plant_tmp__dxr57_1: int, IF_0_25: bool):
    return np.where(IF_0_25, plant_tmp__dxr57_1, plant_tmp__dxr57_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__canht__1(plant_tmp__canht_0: int, plant_tmp__canht_1: int, IF_0_25: bool):
    return np.where(IF_0_25, plant_tmp__canht_1, plant_tmp__canht_0)

def mini_ModuleDefs__put_real__decision__err__6(err_0: bool, err_1: bool, IF_0_25: bool):
    return np.where(IF_0_25, err_1, err_0)

def mini_ModuleDefs__put_real__decision__biomas__1(biomas_0: Real, biomas_1: Real, IF_0_25: bool):
    return np.where(IF_0_25, biomas_1, biomas_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__rnitp__1(plant_tmp__rnitp_0: int, plant_tmp__rnitp_1: int, IF_0_25: bool):
    return np.where(IF_0_25, plant_tmp__rnitp_1, plant_tmp__rnitp_0)

def mini_ModuleDefs__put_real__decision__excess__1(excess_0: Real, excess_1: Real, IF_0_25: bool):
    return np.where(IF_0_25, excess_1, excess_0)

def mini_ModuleDefs__put_real__decision__canwh__1(canwh_0: int, canwh_1: int, IF_0_25: bool):
    return np.where(IF_0_25, canwh_1, canwh_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__biomas__1(plant_tmp__biomas_0: int, plant_tmp__biomas_1: int, IF_0_25: bool):
    return np.where(IF_0_25, plant_tmp__biomas_1, plant_tmp__biomas_0)

def mini_ModuleDefs__put_real__decision__canht__1(canht_0: int, canht_1: int, IF_0_25: bool):
    return np.where(IF_0_25, canht_1, canht_0)

def mini_ModuleDefs__put_real__decision__IF_0__35(IF_0_0: bool, IF_0_1: bool, IF_0_25: bool):
    return np.where(IF_0_25, IF_0_1, IF_0_0)

def mini_ModuleDefs__put_real__decision__slaad__1(slaad_0: Real, slaad_1: Real, IF_0_25: bool):
    return np.where(IF_0_25, slaad_1, slaad_0)

def mini_ModuleDefs__put_real__decision__plant_tmp__excess__1(plant_tmp__excess_0: int, plant_tmp__excess_1: int, IF_0_25: bool):
    return np.where(IF_0_25, plant_tmp__excess_1, plant_tmp__excess_0)

def mini_ModuleDefs__put_real__condition__IF_0__36(varname):
    return (varname == "mgmt")

def mini_ModuleDefs__put_real__assign__mgmt_tmp___1(mgmt: mgmttype):
    return mgmt

def mini_ModuleDefs__put_real__condition__IF_0__37(varname):
    return (varname == "effirr")

def mini_ModuleDefs__put_real__assign__mgmt_tmp__effirr___1(value: Real, mgmt_tmp: mgmttype):
    mgmt_tmp.effirr = value
    return mgmt_tmp.effirr

def mini_ModuleDefs__put_real__decision__effirr__0(effirr_0: Real, effirr_1: Real, IF_0_37: bool):
    return np.where(IF_0_37, effirr_1, effirr_0)

def mini_ModuleDefs__put_real__decision__mgmt_tmp__effirr__0(mgmt_tmp__effirr_0: int, mgmt_tmp__effirr_1: int, IF_0_37: bool):
    return np.where(IF_0_37, mgmt_tmp__effirr_1, mgmt_tmp__effirr_0)

def mini_ModuleDefs__put_real__condition__IF_0__38(varname):
    return (varname == "totir")

def mini_ModuleDefs__put_real__assign__mgmt_tmp__totir___1(value: Real, mgmt_tmp: mgmttype):
    mgmt_tmp.totir = value
    return mgmt_tmp.totir

def mini_ModuleDefs__put_real__decision__totir__0(totir_0: Real, totir_1: Real, IF_0_38: bool):
    return np.where(IF_0_38, totir_1, totir_0)

def mini_ModuleDefs__put_real__decision__mgmt_tmp__totir__0(mgmt_tmp__totir_0: int, mgmt_tmp__totir_1: int, IF_0_38: bool):
    return np.where(IF_0_38, mgmt_tmp__totir_1, mgmt_tmp__totir_0)

def mini_ModuleDefs__put_real__condition__IF_0__39(varname):
    return (varname == "toteffirr")

def mini_ModuleDefs__put_real__assign__mgmt_tmp__toteffirr___1(value: Real, mgmt_tmp: mgmttype):
    mgmt_tmp.toteffirr = value
    return mgmt_tmp.toteffirr

def mini_ModuleDefs__put_real__decision__mgmt_tmp__toteffirr__0(mgmt_tmp__toteffirr_0: int, mgmt_tmp__toteffirr_1: int, IF_0_39: bool):
    return np.where(IF_0_39, mgmt_tmp__toteffirr_1, mgmt_tmp__toteffirr_0)

def mini_ModuleDefs__put_real__decision__toteffirr__0(toteffirr_0: Real, toteffirr_1: Real, IF_0_39: bool):
    return np.where(IF_0_39, toteffirr_1, toteffirr_0)

def mini_ModuleDefs__put_real__condition__IF_0__40(varname):
    return (varname == "depir")

def mini_ModuleDefs__put_real__assign__mgmt_tmp__depir___1(value: Real, mgmt_tmp: mgmttype):
    mgmt_tmp.depir = value
    return mgmt_tmp.depir

def mini_ModuleDefs__put_real__decision__depir__0(depir_0: Array, depir_1: Array, IF_0_40: bool):
    return np.where(IF_0_40, depir_1, depir_0)

def mini_ModuleDefs__put_real__decision__mgmt_tmp__depir__0(mgmt_tmp__depir_0: Array, mgmt_tmp__depir_1: Array, IF_0_40: bool):
    return np.where(IF_0_40, mgmt_tmp__depir_1, mgmt_tmp__depir_0)

def mini_ModuleDefs__put_real__condition__IF_0__41(varname):
    return (varname == "irramt")

def mini_ModuleDefs__put_real__assign__mgmt_tmp__irramt___1(value: Real, mgmt_tmp: mgmttype):
    mgmt_tmp.irramt = value
    return mgmt_tmp.irramt

def mini_ModuleDefs__put_real__decision__mgmt_tmp__irramt__0(mgmt_tmp__irramt_0: int, mgmt_tmp__irramt_1: int, IF_0_41: bool):
    return np.where(IF_0_41, mgmt_tmp__irramt_1, mgmt_tmp__irramt_0)

def mini_ModuleDefs__put_real__decision__irramt__0(irramt_0: Real, irramt_1: Real, IF_0_41: bool):
    return np.where(IF_0_41, irramt_1, irramt_0)

def mini_ModuleDefs__put_real__condition__IF_0__42(varname):
    return (varname == "fernit")

def mini_ModuleDefs__put_real__assign__mgmt_tmp__fernit___1(value: Real, mgmt_tmp: mgmttype):
    mgmt_tmp.fernit = value
    return mgmt_tmp.fernit

def mini_ModuleDefs__put_real__assign__err__7():
    return True

def mini_ModuleDefs__put_real__decision__fernit__0(fernit_0: int, fernit_1: int, IF_0_42: bool):
    return np.where(IF_0_42, fernit_1, fernit_0)

def mini_ModuleDefs__put_real__decision__err__8(err_0: bool, err_1: bool, IF_0_42: bool):
    return np.where(IF_0_42, err_1, err_0)

def mini_ModuleDefs__put_real__decision__mgmt_tmp__fernit__0(mgmt_tmp__fernit_0: int, mgmt_tmp__fernit_1: int, IF_0_42: bool):
    return np.where(IF_0_42, mgmt_tmp__fernit_1, mgmt_tmp__fernit_0)

def mini_ModuleDefs__put_real__decision__mgmt__0(mgmt_0: mgmttype, mgmt_1: mgmttype, IF_0_36: bool):
    return np.where(IF_0_36, mgmt_1, mgmt_0)

def mini_ModuleDefs__put_real__decision__mgmt_tmp__irramt__1(mgmt_tmp__irramt_0: int, mgmt_tmp__irramt_1: int, IF_0_36: bool):
    return np.where(IF_0_36, mgmt_tmp__irramt_1, mgmt_tmp__irramt_0)

def mini_ModuleDefs__put_real__decision__effirr__1(effirr_0: Real, effirr_1: Real, IF_0_36: bool):
    return np.where(IF_0_36, effirr_1, effirr_0)

def mini_ModuleDefs__put_real__decision__irramt__1(irramt_0: Real, irramt_1: Real, IF_0_36: bool):
    return np.where(IF_0_36, irramt_1, irramt_0)

def mini_ModuleDefs__put_real__decision__mgmt_tmp__effirr__1(mgmt_tmp__effirr_0: int, mgmt_tmp__effirr_1: int, IF_0_36: bool):
    return np.where(IF_0_36, mgmt_tmp__effirr_1, mgmt_tmp__effirr_0)

def mini_ModuleDefs__put_real__decision__depir__1(depir_0: Array, depir_1: Array, IF_0_36: bool):
    return np.where(IF_0_36, depir_1, depir_0)

def mini_ModuleDefs__put_real__decision__mgmt_tmp__depir__1(mgmt_tmp__depir_0: Array, mgmt_tmp__depir_1: Array, IF_0_36: bool):
    return np.where(IF_0_36, mgmt_tmp__depir_1, mgmt_tmp__depir_0)

def mini_ModuleDefs__put_real__decision__fernit__1(fernit_0: int, fernit_1: int, IF_0_36: bool):
    return np.where(IF_0_36, fernit_1, fernit_0)

def mini_ModuleDefs__put_real__decision__err__9(err_0: bool, err_1: bool, IF_0_36: bool):
    return np.where(IF_0_36, err_1, err_0)

def mini_ModuleDefs__put_real__decision__totir__1(totir_0: Real, totir_1: Real, IF_0_36: bool):
    return np.where(IF_0_36, totir_1, totir_0)

def mini_ModuleDefs__put_real__decision__mgmt_tmp__toteffirr__1(mgmt_tmp__toteffirr_0: int, mgmt_tmp__toteffirr_1: int, IF_0_36: bool):
    return np.where(IF_0_36, mgmt_tmp__toteffirr_1, mgmt_tmp__toteffirr_0)

def mini_ModuleDefs__put_real__decision__mgmt_tmp__totir__1(mgmt_tmp__totir_0: int, mgmt_tmp__totir_1: int, IF_0_36: bool):
    return np.where(IF_0_36, mgmt_tmp__totir_1, mgmt_tmp__totir_0)

def mini_ModuleDefs__put_real__decision__mgmt_tmp__fernit__1(mgmt_tmp__fernit_0: int, mgmt_tmp__fernit_1: int, IF_0_36: bool):
    return np.where(IF_0_36, mgmt_tmp__fernit_1, mgmt_tmp__fernit_0)

def mini_ModuleDefs__put_real__decision__IF_0__43(IF_0_0: bool, IF_0_1: bool, IF_0_36: bool):
    return np.where(IF_0_36, IF_0_1, IF_0_0)

def mini_ModuleDefs__put_real__decision__toteffirr__1(toteffirr_0: Real, toteffirr_1: Real, IF_0_36: bool):
    return np.where(IF_0_36, toteffirr_1, toteffirr_0)

def mini_ModuleDefs__put_real__condition__IF_0__44(varname):
    return (varname == "water")

def mini_ModuleDefs__put_real__assign__wat_tmp___1(water: wattype):
    return water

def mini_ModuleDefs__put_real__condition__IF_0__45(varname):
    return (varname == "drain")

def mini_ModuleDefs__put_real__assign__wat_tmp__drain___1(value: Real, wat_tmp: wattype):
    wat_tmp.drain = value
    return wat_tmp.drain

def mini_ModuleDefs__put_real__decision__drain__0(drain_0: Real, drain_1: Real, IF_0_45: bool):
    return np.where(IF_0_45, drain_1, drain_0)

def mini_ModuleDefs__put_real__decision__wat_tmp__drain__0(wat_tmp__drain_0: Real, wat_tmp__drain_1: Real, IF_0_45: bool):
    return np.where(IF_0_45, wat_tmp__drain_1, wat_tmp__drain_0)

def mini_ModuleDefs__put_real__condition__IF_0__46(varname):
    return (varname == "runoff")

def mini_ModuleDefs__put_real__assign__wat_tmp__runoff___1(value: Real, wat_tmp: wattype):
    wat_tmp.runoff = value
    return wat_tmp.runoff

def mini_ModuleDefs__put_real__decision__wat_tmp__runoff__0(wat_tmp__runoff_0: int, wat_tmp__runoff_1: int, IF_0_46: bool):
    return np.where(IF_0_46, wat_tmp__runoff_1, wat_tmp__runoff_0)

def mini_ModuleDefs__put_real__decision__runoff__0(runoff_0: int, runoff_1: int, IF_0_46: bool):
    return np.where(IF_0_46, runoff_1, runoff_0)

def mini_ModuleDefs__put_real__condition__IF_0__47(varname):
    return (varname == "snow")

def mini_ModuleDefs__put_real__assign__wat_tmp__snow___1(value: Real, wat_tmp: wattype):
    wat_tmp.snow = value
    return wat_tmp.snow

def mini_ModuleDefs__put_real__assign__err__10():
    return True

def mini_ModuleDefs__put_real__decision__snow__0(snow_0: int, snow_1: int, IF_0_47: bool):
    return np.where(IF_0_47, snow_1, snow_0)

def mini_ModuleDefs__put_real__decision__wat_tmp__snow__0(wat_tmp__snow_0: int, wat_tmp__snow_1: int, IF_0_47: bool):
    return np.where(IF_0_47, wat_tmp__snow_1, wat_tmp__snow_0)

def mini_ModuleDefs__put_real__decision__err__11(err_0: bool, err_1: bool, IF_0_47: bool):
    return np.where(IF_0_47, err_1, err_0)

def mini_ModuleDefs__put_real__decision__drain__1(drain_0: Real, drain_1: Real, IF_0_44: bool):
    return np.where(IF_0_44, drain_1, drain_0)

def mini_ModuleDefs__put_real__decision__wat_tmp__runoff__1(wat_tmp__runoff_0: int, wat_tmp__runoff_1: int, IF_0_44: bool):
    return np.where(IF_0_44, wat_tmp__runoff_1, wat_tmp__runoff_0)

def mini_ModuleDefs__put_real__decision__snow__1(snow_0: int, snow_1: int, IF_0_44: bool):
    return np.where(IF_0_44, snow_1, snow_0)

def mini_ModuleDefs__put_real__decision__wat_tmp__snow__1(wat_tmp__snow_0: int, wat_tmp__snow_1: int, IF_0_44: bool):
    return np.where(IF_0_44, wat_tmp__snow_1, wat_tmp__snow_0)

def mini_ModuleDefs__put_real__decision__err__12(err_0: bool, err_1: bool, IF_0_44: bool):
    return np.where(IF_0_44, err_1, err_0)

def mini_ModuleDefs__put_real__decision__water__0(water_0: wattype, water_1: wattype, IF_0_44: bool):
    return np.where(IF_0_44, water_1, water_0)

def mini_ModuleDefs__put_real__decision__wat_tmp__drain__1(wat_tmp__drain_0: Real, wat_tmp__drain_1: Real, IF_0_44: bool):
    return np.where(IF_0_44, wat_tmp__drain_1, wat_tmp__drain_0)

def mini_ModuleDefs__put_real__decision__IF_0__48(IF_0_0: bool, IF_0_1: bool, IF_0_44: bool):
    return np.where(IF_0_44, IF_0_1, IF_0_0)

def mini_ModuleDefs__put_real__decision__runoff__1(runoff_0: int, runoff_1: int, IF_0_44: bool):
    return np.where(IF_0_44, runoff_1, runoff_0)

def mini_ModuleDefs__put_real__condition__IF_0__49(varname):
    return (varname == "nitr")

def mini_ModuleDefs__put_real__assign__ni_tmp___1(nitr: nitype):
    return nitr

def mini_ModuleDefs__put_real__condition__IF_0__50(varname):
    return (varname == "tnoxd")

def mini_ModuleDefs__put_real__assign__ni_tmp__tnoxd___1(value: Real, ni_tmp: nitype):
    ni_tmp.tnoxd = value
    return ni_tmp.tnoxd

def mini_ModuleDefs__put_real__decision__tnoxd__0(tnoxd_0: int, tnoxd_1: int, IF_0_50: bool):
    return np.where(IF_0_50, tnoxd_1, tnoxd_0)

def mini_ModuleDefs__put_real__decision__ni_tmp__tnoxd__0(ni_tmp__tnoxd_0: int, ni_tmp__tnoxd_1: int, IF_0_50: bool):
    return np.where(IF_0_50, ni_tmp__tnoxd_1, ni_tmp__tnoxd_0)

def mini_ModuleDefs__put_real__condition__IF_0__51(varname):
    return (varname == "tlchd")

def mini_ModuleDefs__put_real__assign__ni_tmp__tleachd___1(value: Real, ni_tmp: nitype):
    ni_tmp.tleachd = value
    return ni_tmp.tleachd

def mini_ModuleDefs__put_real__assign__err__13():
    return True

def mini_ModuleDefs__put_real__decision__ni_tmp__tleachd__0(ni_tmp__tleachd_0: int, ni_tmp__tleachd_1: int, IF_0_51: bool):
    return np.where(IF_0_51, ni_tmp__tleachd_1, ni_tmp__tleachd_0)

def mini_ModuleDefs__put_real__decision__tleachd__0(tleachd_0: Real, tleachd_1: Real, IF_0_51: bool):
    return np.where(IF_0_51, tleachd_1, tleachd_0)

def mini_ModuleDefs__put_real__decision__err__14(err_0: bool, err_1: bool, IF_0_51: bool):
    return np.where(IF_0_51, err_1, err_0)

def mini_ModuleDefs__put_real__decision__ni_tmp__tleachd__1(ni_tmp__tleachd_0: int, ni_tmp__tleachd_1: int, IF_0_49: bool):
    return np.where(IF_0_49, ni_tmp__tleachd_1, ni_tmp__tleachd_0)

def mini_ModuleDefs__put_real__decision__nitr__0(nitr_0: int, nitr_1: int, IF_0_49: bool):
    return np.where(IF_0_49, nitr_1, nitr_0)

def mini_ModuleDefs__put_real__decision__tleachd__1(tleachd_0: Real, tleachd_1: Real, IF_0_49: bool):
    return np.where(IF_0_49, tleachd_1, tleachd_0)

def mini_ModuleDefs__put_real__decision__err__15(err_0: bool, err_1: bool, IF_0_49: bool):
    return np.where(IF_0_49, err_1, err_0)

def mini_ModuleDefs__put_real__decision__tnoxd__1(tnoxd_0: int, tnoxd_1: int, IF_0_49: bool):
    return np.where(IF_0_49, tnoxd_1, tnoxd_0)

def mini_ModuleDefs__put_real__decision__ni_tmp__tnoxd__1(ni_tmp__tnoxd_0: int, ni_tmp__tnoxd_1: int, IF_0_49: bool):
    return np.where(IF_0_49, ni_tmp__tnoxd_1, ni_tmp__tnoxd_0)

def mini_ModuleDefs__put_real__decision__IF_0__52(IF_0_0: bool, IF_0_1: bool, IF_0_49: bool):
    return np.where(IF_0_49, IF_0_1, IF_0_0)

def mini_ModuleDefs__put_real__condition__IF_0__53(varname):
    return (varname == "orgc")

def mini_ModuleDefs__put_real__assign__orgc_tmp___1(orgc: orgctype):
    return orgc

def mini_ModuleDefs__put_real__condition__IF_0__54(varname):
    return (varname == "mulchmass")

def mini_ModuleDefs__put_real__assign__orgc_tmp__mulchmass___1(value: Real, orgc_tmp: orgctype):
    orgc_tmp.mulchmass = value
    return orgc_tmp.mulchmass

def mini_ModuleDefs__put_real__decision__mulchmass__0(mulchmass_0: Real, mulchmass_1: Real, IF_0_54: bool):
    return np.where(IF_0_54, mulchmass_1, mulchmass_0)

def mini_ModuleDefs__put_real__decision__orgc_tmp__mulchmass__0(orgc_tmp__mulchmass_0: Real, orgc_tmp__mulchmass_1: Real, IF_0_54: bool):
    return np.where(IF_0_54, orgc_tmp__mulchmass_1, orgc_tmp__mulchmass_0)

def mini_ModuleDefs__put_real__condition__IF_0__55(varname):
    return (varname == "tominfom")

def mini_ModuleDefs__put_real__assign__orgc_tmp__tominfom___1(value: Real, orgc_tmp: orgctype):
    orgc_tmp.tominfom = value
    return orgc_tmp.tominfom

def mini_ModuleDefs__put_real__decision__tominfom__0(tominfom_0: int, tominfom_1: int, IF_0_55: bool):
    return np.where(IF_0_55, tominfom_1, tominfom_0)

def mini_ModuleDefs__put_real__decision__orgc_tmp__tominfom__0(orgc_tmp__tominfom_0: int, orgc_tmp__tominfom_1: int, IF_0_55: bool):
    return np.where(IF_0_55, orgc_tmp__tominfom_1, orgc_tmp__tominfom_0)

def mini_ModuleDefs__put_real__condition__IF_0__56(varname):
    return (varname == "tominsom")

def mini_ModuleDefs__put_real__assign__orgc_tmp__tominsom___1(value: Real, orgc_tmp: orgctype):
    orgc_tmp.tominsom = value
    return orgc_tmp.tominsom

def mini_ModuleDefs__put_real__decision__orgc_tmp__tominsom__0(orgc_tmp__tominsom_0: int, orgc_tmp__tominsom_1: int, IF_0_56: bool):
    return np.where(IF_0_56, orgc_tmp__tominsom_1, orgc_tmp__tominsom_0)

def mini_ModuleDefs__put_real__decision__tominsom__0(tominsom_0: int, tominsom_1: int, IF_0_56: bool):
    return np.where(IF_0_56, tominsom_1, tominsom_0)

def mini_ModuleDefs__put_real__condition__IF_0__57(varname):
    return (varname == "tominsom1")

def mini_ModuleDefs__put_real__assign__orgc_tmp__tominsom1___1(value: Real, orgc_tmp: orgctype):
    orgc_tmp.tominsom1 = value
    return orgc_tmp.tominsom1

def mini_ModuleDefs__put_real__decision__orgc_tmp__tominsom1__0(orgc_tmp__tominsom1_0: int, orgc_tmp__tominsom1_1: int, IF_0_57: bool):
    return np.where(IF_0_57, orgc_tmp__tominsom1_1, orgc_tmp__tominsom1_0)

def mini_ModuleDefs__put_real__decision__tominsom1__0(tominsom1_0: int, tominsom1_1: int, IF_0_57: bool):
    return np.where(IF_0_57, tominsom1_1, tominsom1_0)

def mini_ModuleDefs__put_real__condition__IF_0__58(varname):
    return (varname == "tominsom2")

def mini_ModuleDefs__put_real__assign__orgc_tmp__tominsom2___1(value: Real, orgc_tmp: orgctype):
    orgc_tmp.tominsom2 = value
    return orgc_tmp.tominsom2

def mini_ModuleDefs__put_real__decision__tominsom2__0(tominsom2_0: int, tominsom2_1: int, IF_0_58: bool):
    return np.where(IF_0_58, tominsom2_1, tominsom2_0)

def mini_ModuleDefs__put_real__decision__orgc_tmp__tominsom2__0(orgc_tmp__tominsom2_0: int, orgc_tmp__tominsom2_1: int, IF_0_58: bool):
    return np.where(IF_0_58, orgc_tmp__tominsom2_1, orgc_tmp__tominsom2_0)

def mini_ModuleDefs__put_real__condition__IF_0__59(varname):
    return (varname == "tominsom3")

def mini_ModuleDefs__put_real__assign__orgc_tmp__tominsom3___1(value: Real, orgc_tmp: orgctype):
    orgc_tmp.tominsom3 = value
    return orgc_tmp.tominsom3

def mini_ModuleDefs__put_real__decision__tominsom3__0(tominsom3_0: int, tominsom3_1: int, IF_0_59: bool):
    return np.where(IF_0_59, tominsom3_1, tominsom3_0)

def mini_ModuleDefs__put_real__decision__orgc_tmp__tominsom3__0(orgc_tmp__tominsom3_0: int, orgc_tmp__tominsom3_1: int, IF_0_59: bool):
    return np.where(IF_0_59, orgc_tmp__tominsom3_1, orgc_tmp__tominsom3_0)

def mini_ModuleDefs__put_real__condition__IF_0__60(varname):
    return (varname == "tnimbsom")

def mini_ModuleDefs__put_real__assign__orgc_tmp__tnimbsom___1(value: Real, orgc_tmp: orgctype):
    orgc_tmp.tnimbsom = value
    return orgc_tmp.tnimbsom

def mini_ModuleDefs__put_real__assign__err__16():
    return True

def mini_ModuleDefs__put_real__decision__err__17(err_0: bool, err_1: bool, IF_0_60: bool):
    return np.where(IF_0_60, err_1, err_0)

def mini_ModuleDefs__put_real__decision__tnimbsom__0(tnimbsom_0: int, tnimbsom_1: int, IF_0_60: bool):
    return np.where(IF_0_60, tnimbsom_1, tnimbsom_0)

def mini_ModuleDefs__put_real__decision__orgc_tmp__tnimbsom__0(orgc_tmp__tnimbsom_0: int, orgc_tmp__tnimbsom_1: int, IF_0_60: bool):
    return np.where(IF_0_60, orgc_tmp__tnimbsom_1, orgc_tmp__tnimbsom_0)

def mini_ModuleDefs__put_real__decision__orgc_tmp__tominsom__1(orgc_tmp__tominsom_0: int, orgc_tmp__tominsom_1: int, IF_0_53: bool):
    return np.where(IF_0_53, orgc_tmp__tominsom_1, orgc_tmp__tominsom_0)

def mini_ModuleDefs__put_real__decision__tominfom__1(tominfom_0: int, tominfom_1: int, IF_0_53: bool):
    return np.where(IF_0_53, tominfom_1, tominfom_0)

def mini_ModuleDefs__put_real__decision__orgc_tmp__tominsom1__1(orgc_tmp__tominsom1_0: int, orgc_tmp__tominsom1_1: int, IF_0_53: bool):
    return np.where(IF_0_53, orgc_tmp__tominsom1_1, orgc_tmp__tominsom1_0)

def mini_ModuleDefs__put_real__decision__tominsom3__1(tominsom3_0: int, tominsom3_1: int, IF_0_53: bool):
    return np.where(IF_0_53, tominsom3_1, tominsom3_0)

def mini_ModuleDefs__put_real__decision__tominsom2__1(tominsom2_0: int, tominsom2_1: int, IF_0_53: bool):
    return np.where(IF_0_53, tominsom2_1, tominsom2_0)

def mini_ModuleDefs__put_real__decision__orgc__0(orgc_0: orgctype, orgc_1: orgctype, IF_0_53: bool):
    return np.where(IF_0_53, orgc_1, orgc_0)

def mini_ModuleDefs__put_real__decision__tominsom1__1(tominsom1_0: int, tominsom1_1: int, IF_0_53: bool):
    return np.where(IF_0_53, tominsom1_1, tominsom1_0)

def mini_ModuleDefs__put_real__decision__mulchmass__1(mulchmass_0: Real, mulchmass_1: Real, IF_0_53: bool):
    return np.where(IF_0_53, mulchmass_1, mulchmass_0)

def mini_ModuleDefs__put_real__decision__err__18(err_0: bool, err_1: bool, IF_0_53: bool):
    return np.where(IF_0_53, err_1, err_0)

def mini_ModuleDefs__put_real__decision__orgc_tmp__mulchmass__1(orgc_tmp__mulchmass_0: Real, orgc_tmp__mulchmass_1: Real, IF_0_53: bool):
    return np.where(IF_0_53, orgc_tmp__mulchmass_1, orgc_tmp__mulchmass_0)

def mini_ModuleDefs__put_real__decision__orgc_tmp__tominsom3__1(orgc_tmp__tominsom3_0: int, orgc_tmp__tominsom3_1: int, IF_0_53: bool):
    return np.where(IF_0_53, orgc_tmp__tominsom3_1, orgc_tmp__tominsom3_0)

def mini_ModuleDefs__put_real__decision__IF_0__61(IF_0_0: bool, IF_0_1: bool, IF_0_53: bool):
    return np.where(IF_0_53, IF_0_1, IF_0_0)

def mini_ModuleDefs__put_real__decision__orgc_tmp__tominfom__1(orgc_tmp__tominfom_0: int, orgc_tmp__tominfom_1: int, IF_0_53: bool):
    return np.where(IF_0_53, orgc_tmp__tominfom_1, orgc_tmp__tominfom_0)

def mini_ModuleDefs__put_real__decision__tominsom__1(tominsom_0: int, tominsom_1: int, IF_0_53: bool):
    return np.where(IF_0_53, tominsom_1, tominsom_0)

def mini_ModuleDefs__put_real__decision__tnimbsom__1(tnimbsom_0: int, tnimbsom_1: int, IF_0_53: bool):
    return np.where(IF_0_53, tnimbsom_1, tnimbsom_0)

def mini_ModuleDefs__put_real__decision__orgc_tmp__tnimbsom__1(orgc_tmp__tnimbsom_0: int, orgc_tmp__tnimbsom_1: int, IF_0_53: bool):
    return np.where(IF_0_53, orgc_tmp__tnimbsom_1, orgc_tmp__tnimbsom_0)

def mini_ModuleDefs__put_real__decision__orgc_tmp__tominsom2__1(orgc_tmp__tominsom2_0: int, orgc_tmp__tominsom2_1: int, IF_0_53: bool):
    return np.where(IF_0_53, orgc_tmp__tominsom2_1, orgc_tmp__tominsom2_0)

def mini_ModuleDefs__put_real__condition__IF_0__62(varname):
    return (varname == "pdlabeta")

def mini_ModuleDefs__put_real__assign__pdlabeta_tmp___1(pdlabeta: pdlabetatype):
    return pdlabeta

def mini_ModuleDefs__put_real__condition__IF_0__63(varname):
    return (varname == "pdla")

def mini_ModuleDefs__put_real__assign__pdlabeta_tmp__pdla___1(value: Real, pdlabeta_tmp: pdlabetatype):
    pdlabeta_tmp.pdla = value
    return pdlabeta_tmp.pdla

def mini_ModuleDefs__put_real__decision__pdla__0(pdla_0: int, pdla_1: int, IF_0_63: bool):
    return np.where(IF_0_63, pdla_1, pdla_0)

def mini_ModuleDefs__put_real__decision__pdlabeta_tmp__pdla__0(pdlabeta_tmp__pdla_0: Real, pdlabeta_tmp__pdla_1: Real, IF_0_63: bool):
    return np.where(IF_0_63, pdlabeta_tmp__pdla_1, pdlabeta_tmp__pdla_0)

def mini_ModuleDefs__put_real__condition__IF_0__64(varname):
    return (varname == "beta")

def mini_ModuleDefs__put_real__assign__pdlabeta_tmp__betals___1(value: Real, pdlabeta_tmp: pdlabetatype):
    pdlabeta_tmp.betals = value
    return pdlabeta_tmp.betals

def mini_ModuleDefs__put_real__assign__err__19():
    return True

def mini_ModuleDefs__put_real__decision__pdlabeta_tmp__betals__0(pdlabeta_tmp__betals_0: Real, pdlabeta_tmp__betals_1: Real, IF_0_64: bool):
    return np.where(IF_0_64, pdlabeta_tmp__betals_1, pdlabeta_tmp__betals_0)

def mini_ModuleDefs__put_real__decision__betals__0(betals_0: Real, betals_1: Real, IF_0_64: bool):
    return np.where(IF_0_64, betals_1, betals_0)

def mini_ModuleDefs__put_real__decision__err__20(err_0: bool, err_1: bool, IF_0_64: bool):
    return np.where(IF_0_64, err_1, err_0)

def mini_ModuleDefs__put_real__assign__err__21():
    return True

def mini_ModuleDefs__put_real__decision__pdlabeta_tmp__betals__1(pdlabeta_tmp__betals_0: Real, pdlabeta_tmp__betals_1: Real, IF_0_62: bool):
    return np.where(IF_0_62, pdlabeta_tmp__betals_1, pdlabeta_tmp__betals_0)

def mini_ModuleDefs__put_real__decision__betals__1(betals_0: Real, betals_1: Real, IF_0_62: bool):
    return np.where(IF_0_62, betals_1, betals_0)

def mini_ModuleDefs__put_real__decision__pdlabeta__0(pdlabeta_0: Real, pdlabeta_1: Real, IF_0_62: bool):
    return np.where(IF_0_62, pdlabeta_1, pdlabeta_0)

def mini_ModuleDefs__put_real__decision__err__22(err_0: bool, err_1: bool, IF_0_62: bool):
    return np.where(IF_0_62, err_1, err_0)

def mini_ModuleDefs__put_real__decision__pdla__1(pdla_0: int, pdla_1: int, IF_0_62: bool):
    return np.where(IF_0_62, pdla_1, pdla_0)

def mini_ModuleDefs__put_real__decision__IF_0__65(IF_0_0: bool, IF_0_1: bool, IF_0_62: bool):
    return np.where(IF_0_62, IF_0_1, IF_0_0)

def mini_ModuleDefs__put_real__decision__pdlabeta_tmp__pdla__1(pdlabeta_tmp__pdla_0: Real, pdlabeta_tmp__pdla_1: Real, IF_0_62: bool):
    return np.where(IF_0_62, pdlabeta_tmp__pdla_1, pdlabeta_tmp__pdla_0)

def mini_ModuleDefs__get_real_array_nl__assign__value__0():
    return 0.0

def mini_ModuleDefs__get_real_array_nl__assign__err__0():
    return False

def mini_ModuleDefs__get_real_array_nl__condition__IF_0__0(modulename):
    return (modulename == "spam")

def mini_ModuleDefs__get_real_array_nl__condition__IF_0__1(varname):
    return (varname == "uh2o")

def mini_ModuleDefs__get_real_array_nl__assign__spam_tmp___1(spam: spamtype):
    return spam

def mini_ModuleDefs__get_real_array_nl__assign__value__1(uh2o: Array):
    return uh2o

def mini_ModuleDefs__get_real_array_nl__assign__err__1():
    return True

def mini_ModuleDefs__get_real_array_nl__decision__value__2(value_0: Real, value_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, value_1, value_0)

def mini_ModuleDefs__get_real_array_nl__decision__spam__0(spam_0: int, spam_1: int, IF_0_1: bool):
    return np.where(IF_0_1, spam_1, spam_0)

def mini_ModuleDefs__get_real_array_nl__decision__uh2o__0(uh2o_0: Real, uh2o_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, uh2o_1, uh2o_0)

def mini_ModuleDefs__get_real_array_nl__decision__err__2(err_0: bool, err_1: bool, IF_0_1: bool):
    return np.where(IF_0_1, err_1, err_0)

def mini_ModuleDefs__get_real_array_nl__assign__err__3():
    return True

def mini_ModuleDefs__get_real_array_nl__decision__value__3(value_0: Real, value_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, value_1, value_0)

def mini_ModuleDefs__get_real_array_nl__decision__spam__1(spam_0: int, spam_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_1, spam_0)

def mini_ModuleDefs__get_real_array_nl__decision__uh2o__1(uh2o_0: Real, uh2o_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, uh2o_1, uh2o_0)

def mini_ModuleDefs__get_real_array_nl__decision__IF_0__2(IF_0_0: bool, IF_0_1: bool):
    return np.where(IF_0_0, IF_0_1, IF_0_0)

def mini_ModuleDefs__get_real_array_nl__decision__err__4(err_0: bool, err_1: bool, IF_0_0: bool):
    return np.where(IF_0_0, err_1, err_0)

def mini_ModuleDefs__put_real_array_nl__assign__err__0():
    return False

def mini_ModuleDefs__put_real_array_nl__condition__IF_0__0(modulename):
    return (modulename == "spam")

def mini_ModuleDefs__put_real_array_nl__condition__IF_0__1(varname):
    return (varname == "uh2o")

def mini_ModuleDefs__put_real_array_nl__assign__save_data__spam__uh2o___1(value: Array, save_data: transfertype):
    save_data.spam.uh2o = value
    return save_data.spam.uh2o

def mini_ModuleDefs__put_real_array_nl__assign__err__1():
    return True

def mini_ModuleDefs__put_real_array_nl__decision__save_data__spam__uh2o__0(save_data__spam__uh2o_0: Real, save_data__spam__uh2o_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, save_data__spam__uh2o_1, save_data__spam__uh2o_0)

def mini_ModuleDefs__put_real_array_nl__decision__spam__0(spam_0: int, spam_1: int, IF_0_1: bool):
    return np.where(IF_0_1, spam_1, spam_0)

def mini_ModuleDefs__put_real_array_nl__decision__uh2o__0(uh2o_0: Real, uh2o_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, uh2o_1, uh2o_0)

def mini_ModuleDefs__put_real_array_nl__decision__err__2(err_0: bool, err_1: bool, IF_0_1: bool):
    return np.where(IF_0_1, err_1, err_0)

def mini_ModuleDefs__put_real_array_nl__assign__err__3():
    return True

def mini_ModuleDefs__put_real_array_nl__decision__save_data__spam__uh2o__1(save_data__spam__uh2o_0: Real, save_data__spam__uh2o_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, save_data__spam__uh2o_1, save_data__spam__uh2o_0)

def mini_ModuleDefs__put_real_array_nl__decision__spam__1(spam_0: int, spam_1: int, IF_0_0: bool):
    return np.where(IF_0_0, spam_1, spam_0)

def mini_ModuleDefs__put_real_array_nl__decision__uh2o__1(uh2o_0: Real, uh2o_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, uh2o_1, uh2o_0)

def mini_ModuleDefs__put_real_array_nl__decision__IF_0__2(IF_0_0: bool, IF_0_1: bool):
    return np.where(IF_0_0, IF_0_1, IF_0_0)

def mini_ModuleDefs__put_real_array_nl__decision__err__4(err_0: bool, err_1: bool, IF_0_0: bool):
    return np.where(IF_0_0, err_1, err_0)

def mini_ModuleDefs__get_integer__assign__value__0():
    return 0

def mini_ModuleDefs__get_integer__assign__err__0():
    return False

def mini_ModuleDefs__get_integer__condition__IF_0__0(modulename):
    return (modulename == "plant")

def mini_ModuleDefs__get_integer__assign__plant_tmp___1(plant: planttype):
    return plant

def mini_ModuleDefs__get_integer__condition__IF_0__1(varname):
    return (varname == "nr5")

def mini_ModuleDefs__get_integer__assign__value__1(nr5: int):
    return nr5

def mini_ModuleDefs__get_integer__decision__value__2(value_0: Real, value_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, value_1, value_0)

def mini_ModuleDefs__get_integer__decision__nr5__0(nr5_0: int, nr5_1: int, IF_0_1: bool):
    return np.where(IF_0_1, nr5_1, nr5_0)

def mini_ModuleDefs__get_integer__condition__IF_0__2(varname):
    return (varname == "istage")

def mini_ModuleDefs__get_integer__assign__value__3(istage: int):
    return istage

def mini_ModuleDefs__get_integer__decision__value__4(value_0: Real, value_1: Real, IF_0_2: bool):
    return np.where(IF_0_2, value_1, value_0)

def mini_ModuleDefs__get_integer__decision__istage__0(istage_0: Real, istage_1: Real, IF_0_2: bool):
    return np.where(IF_0_2, istage_1, istage_0)

def mini_ModuleDefs__get_integer__condition__IF_0__3(varname):
    return (varname == "istgdoy")

def mini_ModuleDefs__get_integer__assign__value__5(istgdoy: int):
    return istgdoy

def mini_ModuleDefs__get_integer__assign__err__1():
    return True

def mini_ModuleDefs__get_integer__decision__value__6(value_0: Real, value_1: Real, IF_0_3: bool):
    return np.where(IF_0_3, value_1, value_0)

def mini_ModuleDefs__get_integer__decision__istgdoy__0(istgdoy_0: int, istgdoy_1: int, IF_0_3: bool):
    return np.where(IF_0_3, istgdoy_1, istgdoy_0)

def mini_ModuleDefs__get_integer__decision__err__2(err_0: bool, err_1: bool, IF_0_3: bool):
    return np.where(IF_0_3, err_1, err_0)

def mini_ModuleDefs__get_integer__assign__err__3():
    return True

def mini_ModuleDefs__get_integer__decision__value__7(value_0: Real, value_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, value_1, value_0)

def mini_ModuleDefs__get_integer__decision__plant__0(plant_0: int, plant_1: int, IF_0_0: bool):
    return np.where(IF_0_0, plant_1, plant_0)

def mini_ModuleDefs__get_integer__decision__nr5__1(nr5_0: int, nr5_1: int, IF_0_0: bool):
    return np.where(IF_0_0, nr5_1, nr5_0)

def mini_ModuleDefs__get_integer__decision__istgdoy__1(istgdoy_0: int, istgdoy_1: int, IF_0_0: bool):
    return np.where(IF_0_0, istgdoy_1, istgdoy_0)

def mini_ModuleDefs__get_integer__decision__istage__1(istage_0: Real, istage_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, istage_1, istage_0)

def mini_ModuleDefs__get_integer__decision__IF_0__4(IF_0_0: bool, IF_0_1: bool):
    return np.where(IF_0_0, IF_0_1, IF_0_0)

def mini_ModuleDefs__get_integer__decision__err__4(err_0: bool, err_1: bool, IF_0_0: bool):
    return np.where(IF_0_0, err_1, err_0)

def mini_ModuleDefs__put_integer__assign__err__0():
    return False

def mini_ModuleDefs__put_integer__condition__IF_0__0(modulename):
    return (modulename == "plant")

def mini_ModuleDefs__put_integer__assign__plant_tmp___1(plant: planttype):
    return plant

def mini_ModuleDefs__put_integer__condition__IF_0__1(varname):
    return (varname == "nr5")

def mini_ModuleDefs__put_integer__assign__plant_tmp__nr5___1(value: int, plant_tmp: planttype):
    plant_tmp.nr5 = value
    return plant_tmp.nr5

def mini_ModuleDefs__put_integer__decision__plant_tmp__nr5__0(plant_tmp__nr5_0: int, plant_tmp__nr5_1: int, IF_0_1: bool):
    return np.where(IF_0_1, plant_tmp__nr5_1, plant_tmp__nr5_0)

def mini_ModuleDefs__put_integer__decision__nr5__0(nr5_0: int, nr5_1: int, IF_0_1: bool):
    return np.where(IF_0_1, nr5_1, nr5_0)

def mini_ModuleDefs__put_integer__condition__IF_0__2(varname):
    return (varname == "istage")

def mini_ModuleDefs__put_integer__assign__plant_tmp__istage___1(value: int, plant_tmp: planttype):
    plant_tmp.istage = value
    return plant_tmp.istage

def mini_ModuleDefs__put_integer__decision__plant_tmp__istage__0(plant_tmp__istage_0: Real, plant_tmp__istage_1: Real, IF_0_2: bool):
    return np.where(IF_0_2, plant_tmp__istage_1, plant_tmp__istage_0)

def mini_ModuleDefs__put_integer__decision__istage__0(istage_0: Real, istage_1: Real, IF_0_2: bool):
    return np.where(IF_0_2, istage_1, istage_0)

def mini_ModuleDefs__put_integer__condition__IF_0__3(varname):
    return (varname == "istgdoy")

def mini_ModuleDefs__put_integer__assign__plant_tmp__istgdoy___1(value: int, plant_tmp: planttype):
    plant_tmp.istgdoy = value
    return plant_tmp.istgdoy

def mini_ModuleDefs__put_integer__assign__err__1():
    return True

def mini_ModuleDefs__put_integer__decision__plant_tmp__istgdoy__0(plant_tmp__istgdoy_0: int, plant_tmp__istgdoy_1: int, IF_0_3: bool):
    return np.where(IF_0_3, plant_tmp__istgdoy_1, plant_tmp__istgdoy_0)

def mini_ModuleDefs__put_integer__decision__istgdoy__0(istgdoy_0: int, istgdoy_1: int, IF_0_3: bool):
    return np.where(IF_0_3, istgdoy_1, istgdoy_0)

def mini_ModuleDefs__put_integer__decision__err__2(err_0: bool, err_1: bool, IF_0_3: bool):
    return np.where(IF_0_3, err_1, err_0)

def mini_ModuleDefs__put_integer__assign__err__3():
    return True

def mini_ModuleDefs__put_integer__decision__plant_tmp__istage__1(plant_tmp__istage_0: Real, plant_tmp__istage_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, plant_tmp__istage_1, plant_tmp__istage_0)

def mini_ModuleDefs__put_integer__decision__plant__0(plant_0: int, plant_1: int, IF_0_0: bool):
    return np.where(IF_0_0, plant_1, plant_0)

def mini_ModuleDefs__put_integer__decision__nr5__1(nr5_0: int, nr5_1: int, IF_0_0: bool):
    return np.where(IF_0_0, nr5_1, nr5_0)

def mini_ModuleDefs__put_integer__decision__plant_tmp__istgdoy__1(plant_tmp__istgdoy_0: int, plant_tmp__istgdoy_1: int, IF_0_0: bool):
    return np.where(IF_0_0, plant_tmp__istgdoy_1, plant_tmp__istgdoy_0)

def mini_ModuleDefs__put_integer__decision__istgdoy__1(istgdoy_0: int, istgdoy_1: int, IF_0_0: bool):
    return np.where(IF_0_0, istgdoy_1, istgdoy_0)

def mini_ModuleDefs__put_integer__decision__istage__1(istage_0: Real, istage_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, istage_1, istage_0)

def mini_ModuleDefs__put_integer__decision__IF_0__4(IF_0_0: bool, IF_0_1: bool):
    return np.where(IF_0_0, IF_0_1, IF_0_0)

def mini_ModuleDefs__put_integer__decision__plant_tmp__nr5__1(plant_tmp__nr5_0: int, plant_tmp__nr5_1: int, IF_0_0: bool):
    return np.where(IF_0_0, plant_tmp__nr5_1, plant_tmp__nr5_0)

def mini_ModuleDefs__put_integer__decision__err__4(err_0: bool, err_1: bool, IF_0_0: bool):
    return np.where(IF_0_0, err_1, err_0)

def mini_ModuleDefs__get_char__assign__value__0():
    return Float32(" ")[0:Undefined].ljust(Undefined, " ")

def mini_ModuleDefs__get_char__assign__err__0():
    return False

def mini_ModuleDefs__get_char__condition__IF_0__0(modulename):
    return (modulename == "weather")

def mini_ModuleDefs__get_char__condition__IF_0__1(varname):
    return (varname == "wsta")

def mini_ModuleDefs__get_char__assign__weath_tmp___1(weather: weathtype):
    return weather

def mini_ModuleDefs__get_char__assign__value__1(wstat: string):
    return Float32(wstat)[0:Undefined].ljust(Undefined, " ")

def mini_ModuleDefs__get_char__assign__err__1():
    return True

def mini_ModuleDefs__get_char__decision__value__2(value_0: Real, value_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, value_1, value_0)

def mini_ModuleDefs__get_char__decision__wstat__0(wstat_0: Real, wstat_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, wstat_1, wstat_0)

def mini_ModuleDefs__get_char__decision__weather__0(weather_0: weathtype, weather_1: weathtype, IF_0_1: bool):
    return np.where(IF_0_1, weather_1, weather_0)

def mini_ModuleDefs__get_char__decision__err__2(err_0: bool, err_1: bool, IF_0_1: bool):
    return np.where(IF_0_1, err_1, err_0)

def mini_ModuleDefs__get_char__decision__value__3(value_0: Real, value_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, value_1, value_0)

def mini_ModuleDefs__get_char__decision__wstat__1(wstat_0: Real, wstat_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, wstat_1, wstat_0)

def mini_ModuleDefs__get_char__decision__weather__1(weather_0: weathtype, weather_1: weathtype, IF_0_0: bool):
    return np.where(IF_0_0, weather_1, weather_0)

def mini_ModuleDefs__get_char__decision__IF_0__2(IF_0_0: bool, IF_0_1: bool):
    return np.where(IF_0_0, IF_0_1, IF_0_0)

def mini_ModuleDefs__get_char__decision__err__3(err_0: bool, err_1: bool, IF_0_0: bool):
    return np.where(IF_0_0, err_1, err_0)

def mini_ModuleDefs__get_char__condition__IF_0__3(varname):
    return (varname == "plant")

def mini_ModuleDefs__get_char__condition__IF_0__4(varname):
    return (varname == "istname")

def mini_ModuleDefs__get_char__assign__plant_tmp___1(plant: planttype):
    return plant

def mini_ModuleDefs__get_char__assign__value__4(istname: string):
    return Float32(istname)[0:Undefined].ljust(Undefined, " ")

def mini_ModuleDefs__get_char__assign__err__4():
    return True

def mini_ModuleDefs__get_char__decision__value__5(value_0: Real, value_1: Real, IF_0_4: bool):
    return np.where(IF_0_4, value_1, value_0)

def mini_ModuleDefs__get_char__decision__plant__0(plant_0: int, plant_1: int, IF_0_4: bool):
    return np.where(IF_0_4, plant_1, plant_0)

def mini_ModuleDefs__get_char__decision__istname__0(istname_0: int, istname_1: int, IF_0_4: bool):
    return np.where(IF_0_4, istname_1, istname_0)

def mini_ModuleDefs__get_char__decision__err__5(err_0: bool, err_1: bool, IF_0_4: bool):
    return np.where(IF_0_4, err_1, err_0)

def mini_ModuleDefs__get_char__assign__err__6():
    return True

def mini_ModuleDefs__get_char__decision__value__6(value_0: Real, value_1: Real, IF_0_3: bool):
    return np.where(IF_0_3, value_1, value_0)

def mini_ModuleDefs__get_char__decision__plant__1(plant_0: int, plant_1: int, IF_0_3: bool):
    return np.where(IF_0_3, plant_1, plant_0)

def mini_ModuleDefs__get_char__decision__istname__1(istname_0: int, istname_1: int, IF_0_3: bool):
    return np.where(IF_0_3, istname_1, istname_0)

def mini_ModuleDefs__get_char__decision__IF_0__5(IF_0_0: bool, IF_0_1: bool, IF_0_3: bool):
    return np.where(IF_0_3, IF_0_1, IF_0_0)

def mini_ModuleDefs__get_char__decision__err__7(err_0: bool, err_1: bool, IF_0_3: bool):
    return np.where(IF_0_3, err_1, err_0)

def mini_ModuleDefs__put_char__assign__err__0():
    return False

def mini_ModuleDefs__put_char__condition__IF_0__0(modulename):
    return (modulename == "weather")

def mini_ModuleDefs__put_char__condition__IF_0__1(varname):
    return (varname == "wsta")

def mini_ModuleDefs__put_char__assign__weath_tmp___1(weather: weathtype):
    return weather

def mini_ModuleDefs__put_char__assign__weath_tmp__wstat___1(value, weath_tmp: weathtype):
    weath_tmp.wstat = value
    return weath_tmp.wstat

def mini_ModuleDefs__put_char__assign__err__1():
    return True

def mini_ModuleDefs__put_char__decision__weath_tmp__wstat__0(weath_tmp__wstat_0: Real, weath_tmp__wstat_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, weath_tmp__wstat_1, weath_tmp__wstat_0)

def mini_ModuleDefs__put_char__decision__wstat__0(wstat_0: Real, wstat_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, wstat_1, wstat_0)

def mini_ModuleDefs__put_char__decision__weather__0(weather_0: weathtype, weather_1: weathtype, IF_0_1: bool):
    return np.where(IF_0_1, weather_1, weather_0)

def mini_ModuleDefs__put_char__decision__err__2(err_0: bool, err_1: bool, IF_0_1: bool):
    return np.where(IF_0_1, err_1, err_0)

def mini_ModuleDefs__put_char__decision__weath_tmp__wstat__1(weath_tmp__wstat_0: Real, weath_tmp__wstat_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, weath_tmp__wstat_1, weath_tmp__wstat_0)

def mini_ModuleDefs__put_char__decision__wstat__1(wstat_0: Real, wstat_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, wstat_1, wstat_0)

def mini_ModuleDefs__put_char__decision__weather__1(weather_0: weathtype, weather_1: weathtype, IF_0_0: bool):
    return np.where(IF_0_0, weather_1, weather_0)

def mini_ModuleDefs__put_char__decision__IF_0__2(IF_0_0: bool, IF_0_1: bool):
    return np.where(IF_0_0, IF_0_1, IF_0_0)

def mini_ModuleDefs__put_char__decision__err__3(err_0: bool, err_1: bool, IF_0_0: bool):
    return np.where(IF_0_0, err_1, err_0)

def mini_ModuleDefs__put_char__condition__IF_0__3(varname):
    return (varname == "plant")

def mini_ModuleDefs__put_char__condition__IF_0__4(varname):
    return (varname == "istname")

def mini_ModuleDefs__put_char__assign__plant_tmp___1(plant: planttype):
    return plant

def mini_ModuleDefs__put_char__assign__plant_tmp__istname___1(value, plant_tmp: planttype):
    plant_tmp.istname = value
    return plant_tmp.istname

def mini_ModuleDefs__put_char__assign__err__4():
    return True

def mini_ModuleDefs__put_char__decision__plant__0(plant_0: int, plant_1: int, IF_0_4: bool):
    return np.where(IF_0_4, plant_1, plant_0)

def mini_ModuleDefs__put_char__decision__istname__0(istname_0: int, istname_1: int, IF_0_4: bool):
    return np.where(IF_0_4, istname_1, istname_0)

def mini_ModuleDefs__put_char__decision__plant_tmp__istname__0(plant_tmp__istname_0: int, plant_tmp__istname_1: int, IF_0_4: bool):
    return np.where(IF_0_4, plant_tmp__istname_1, plant_tmp__istname_0)

def mini_ModuleDefs__put_char__decision__err__5(err_0: bool, err_1: bool, IF_0_4: bool):
    return np.where(IF_0_4, err_1, err_0)

def mini_ModuleDefs__put_char__assign__err__6():
    return True

def mini_ModuleDefs__put_char__decision__plant__1(plant_0: int, plant_1: int, IF_0_3: bool):
    return np.where(IF_0_3, plant_1, plant_0)

def mini_ModuleDefs__put_char__decision__istname__1(istname_0: int, istname_1: int, IF_0_3: bool):
    return np.where(IF_0_3, istname_1, istname_0)

def mini_ModuleDefs__put_char__decision__IF_0__5(IF_0_0: bool, IF_0_1: bool, IF_0_3: bool):
    return np.where(IF_0_3, IF_0_1, IF_0_0)

def mini_ModuleDefs__put_char__decision__plant_tmp__istname__1(plant_tmp__istname_0: int, plant_tmp__istname_1: int, IF_0_3: bool):
    return np.where(IF_0_3, plant_tmp__istname_1, plant_tmp__istname_0)

def mini_ModuleDefs__put_char__decision__err__7(err_0: bool, err_1: bool, IF_0_3: bool):
    return np.where(IF_0_3, err_1, err_0)

