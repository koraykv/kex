require 'lab'
require 'nn'
require 'libkex'

torch.include('kex','lushio.lua')
-- extra modules that we need
torch.include('kex', 'SpatialFullConvolution.lua')
torch.include('kex', 'WeightedMSECriterion.lua')
torch.include('kex', 'L1Cost.lua')
torch.include('kex', 'CriterionModule.lua')
torch.include('kex', 'stochasticrates.lua')
