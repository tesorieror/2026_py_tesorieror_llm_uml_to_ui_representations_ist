import arviz as az

idata = az.from_netcdf(r'experiment/analysis/results/generation_success_beta_binomial/posterior.nc')

s = az.summary(idata,var_names=['intercept','model_effect_raw','representation_effect_raw','interaction_raw','sigma_uml','uml_raw','overdispersion'],round_to=None)

print('max_rhat:',s['r_hat'].max())
print('min_ess_bulk:',s['ess_bulk'].min())
print('min_ess_tail:',s['ess_tail'].min())
print(s.loc[[i for i in s.index if i.startswith('uml_raw')],['r_hat','ess_bulk','ess_tail']])