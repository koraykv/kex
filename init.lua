require 'torch'
require 'nn'
require 'libkex'

torch.include('kex','lushio.lua')
-- extra modules that we need
torch.include('kex', 'SpatialFullConvolution.lua')
torch.include('kex', 'WeightedMSECriterion.lua')
torch.include('kex', 'L1Cost.lua')
torch.include('kex', 'Diag.lua')
torch.include('kex', 'CriterionModule.lua')
torch.include('kex', 'SpatialMaxPyramid.lua')
torch.include('kex', 'SpatialMaxPooling2.lua')
torch.include('kex', 'Vectorize.lua')
torch.include('kex', 'TanhShrink.lua')
torch.include('kex', 'Crop.lua')
torch.include('kex', 'SqrtBias.lua')
torch.include('kex', 'stochasticrates.lua')
torch.include('kex', 'TensorLinear.lua')

function kex.cudahacks()
   torch.include('kex','cudahacks.lua')
end
